from enum import IntEnum
import math
from typing import Any, Optional, Tuple, Union, Dict
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as onp
from flax.struct import dataclass
import colorsys

from opensocialjax.environments.movement import resolve_movement, resolve_respawn
from opensocialjax.environments.multi_agent_env import MultiAgentEnv
from opensocialjax.environments import spaces
from opensocialjax.environments.discovery_rules import (
    CRAFT_SEQUENCE,
    MAX_RULES,
    NO_EFFECT,
    RULE_ENCODING_LEN,
    TOOL_CLEARS,
    TOOL_TRANSFORMS,
    apply_beam,
    check_clean,
    decode_attr_tables,
    decode_chain,
    decode_craft_recipes,
    decode_craft_sequence,
    decode_tool_table,
    encode_craft_sequence,
    encode_tool_table,
    pad_ruleset,
    sample_attr_craft_ruleset,
    sample_type_craft_ruleset,
    sample_attr_ruleset,
    sample_craft_ruleset,
)


from opensocialjax.environments.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_line,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

NUM_TYPES = 4  # empty (0), red (1), blue, red coin, blue coin, wall, interact
NUM_COIN_TYPES = 1
INTERACT_THRESHOLD = 0


@dataclass
class State:
    agent_locs: jnp.ndarray
    agent_invs: jnp.ndarray
    inner_t: int
    outer_t: int
    grid: jnp.ndarray

    apples: jnp.ndarray
    freeze: jnp.ndarray
    reborn_locs: jnp.ndarray

    potential_dirt_and_dirt_locs: jnp.ndarray
    potential_dirt_and_dirt_label: jnp.ndarray
    smooth_rewards: jnp.ndarray
    # Hidden ruleset for this episode, drawn at reset and never shown to the
    # agents. XLand-style encoding: (MAX_RULES, RULE_ENCODING_LEN) uint8 rows
    # of [rule_id, args...], zero rows being EmptyRule padding. The step
    # function interprets it with discovery_rules.check_clean; nothing decodes
    # it into Python-level branches.
    rule_encoding: jnp.ndarray
    # Which tool each agent is carrying, as an index into TOOL_TYPES.
    held_tool: jnp.ndarray
    # This trial's dominant waste type (index 0..NUM_DIRT_TYPES-1) — or its
    # dominant HUE (0..NUM_HUES-1) in the attribute variant — redrawn at every
    # trial reset. Only consulted when skewed_waste is on; kept in
    # State unconditionally so the pytree structure never depends on flags.
    dom_type: jnp.ndarray
    # Crafting variant: each agent's last craft_len station pickups (-1 =
    # none yet), oldest first, and whether the agent holds THE cleaning tool.
    # Kept in State unconditionally; zeros/-1 outside the variant.
    pickups: jnp.ndarray
    crafted: jnp.ndarray
    # Which map of the layout pool this meta-episode plays on (0 without a pool); kept across its trials.
    layout_id: jnp.ndarray


@chex.dataclass
class EnvParams:
    payoff_matrix: chex.ArrayDevice
    freeze_penalty: int

class Actions(IntEnum):
    turn_left = 0
    turn_right = 1
    left = 2
    right = 3
    up = 4
    down = 5
    stay = 6
    zap_forward = 7
    zap_clean = 8
    pick_up = 9      # only with explicit_pickup=True: take the tool of the station you stand on


class Items(IntEnum):
    empty = 0
    wall = 1
    interact = 2
    apple = 3
    spawn_point = 4
    inside_spawn_point = 5
    river = 6
    potential_dirt = 7
    # Coloured kinds, XLand style. A waste cell or a tool is a (kind, colour)
    # pair exactly like xland-minigrid's (tile, color): DIRT in one of
    # NUM_COLORS colours, TOOL in one of NUM_COLORS colours. The grid stays a
    # single int16 layer, so every (kind, colour) pair gets a code from the
    # ITEM_CODE registry below (the analogue of TILES_REGISTRY); this enum
    # materialises those codes so len(Items) still counts every item code
    # (agents are numbered from len(Items) on). All waste colours pollute,
    # exactly as the single waste type does in vanilla Clean Up — what is
    # hidden is which TOOL colour clears which WASTE colour.
    # Fifteen waste colours x three tools = 3**15 = 14,348,907 rules. Three tools
    # (not four) on purpose: what sets the per-task difficulty is the share
    # of the river one tool can clear, 1/NUM_TOOL_COLORS — the design gate
    # showed a fourth tool drops that to 1/4 and pushes cleaning throughput
    # below the pollution threshold (oracle -69%, cycle/blind to zero).
    dirt_a = 8
    dirt_b = 9
    dirt_c = 10
    dirt_d = 11
    dirt_e = 12
    dirt_f = 13
    dirt_g = 14
    dirt_h = 15
    dirt_i = 16
    dirt_j = 17
    dirt_k = 18
    dirt_l = 19
    dirt_m = 20
    dirt_n = 21
    dirt_o = 22
    # Cleaning tools lying on the floor. Stepping on one swaps the agent's
    # held tool and leaves the cell in place, so a tool is a choice to make
    # rather than a resource to ration. The tool block spans the whole
    # colour axis; only the first NUM_TOOL_COLORS colours are placed.
    tool_a = 23
    tool_b = 24
    tool_c = 25
    tool_d = 26
    tool_e = 27
    tool_f = 28
    tool_g = 29
    tool_h = 30
    tool_i = 31
    tool_j = 32
    tool_k = 33
    tool_l = 34
    tool_m = 35
    tool_n = 36
    tool_o = 37
    clean_beam = 38


# ---------------------------------------------------------------------------
# (kind, colour) registry — the xland-minigrid representation, on a single
# grid layer. Kinds are the item "shapes"; colours are a shared axis that
# both DIRT and TOOL draw from (not every kind uses every colour, as in
# XLand). Widening the task space is changing NUM_DIRT_COLORS /
# NUM_TOOL_COLORS: the enum blocks, the registry, all_rules(), the benchmark
# split, the observation and the renderer all follow.
# ---------------------------------------------------------------------------
NUM_COLORS = 15           # size of the shared colour axis
NUM_DIRT_COLORS = 15      # colours waste comes in         (rule table rows)
NUM_TOOL_COLORS = 3       # colours tools come in          (rule table values)
assert NUM_DIRT_COLORS <= NUM_COLORS and NUM_TOOL_COLORS <= NUM_COLORS


class Kinds(IntEnum):
    empty = 0
    wall = 1
    interact = 2
    apple = 3
    spawn_point = 4
    inside_spawn_point = 5
    river = 6
    potential_dirt = 7
    dirt = 8
    tool = 9
    beam = 10


NUM_KINDS = len(Kinds)
DIRT_BASE = int(Items.dirt_a)
TOOL_BASE = int(Items.tool_a)
assert TOOL_BASE == DIRT_BASE + NUM_COLORS and int(Items.clean_beam) == TOOL_BASE + NUM_COLORS

# ITEM_CODE[kind, colour] -> grid code. Uncoloured kinds map to their single
# code for every colour, so make_item(kind, colour) is total.
_codes = onp.zeros((NUM_KINDS, NUM_COLORS), dtype=onp.int16)
for _k in range(NUM_KINDS):
    _codes[_k, :] = _k
_codes[Kinds.dirt, :] = DIRT_BASE + onp.arange(NUM_COLORS)
_codes[Kinds.tool, :] = TOOL_BASE + onp.arange(NUM_COLORS)
_codes[Kinds.beam, :] = int(Items.clean_beam)
ITEM_CODE = jnp.asarray(_codes)

# inverse lookups over item codes (agents, numbered from len(Items), clip to
# the last entry — callers never ask for an agent's kind or colour)
_kind_of = onp.arange(len(Items), dtype=onp.int16)
_kind_of[DIRT_BASE:DIRT_BASE + NUM_COLORS] = Kinds.dirt
_kind_of[TOOL_BASE:TOOL_BASE + NUM_COLORS] = Kinds.tool
_kind_of[int(Items.clean_beam)] = Kinds.beam
_color_of = onp.zeros(len(Items), dtype=onp.int16)
_color_of[DIRT_BASE:DIRT_BASE + NUM_COLORS] = onp.arange(NUM_COLORS)
_color_of[TOOL_BASE:TOOL_BASE + NUM_COLORS] = onp.arange(NUM_COLORS)
KIND_OF = jnp.asarray(_kind_of)
COLOR_OF = jnp.asarray(_color_of)


def make_item(kind, color):
    """Grid code of a (kind, colour) pair — xland's TILES_REGISTRY[tile, color]."""
    return ITEM_CODE[kind, color]


def kind_of(code):
    return KIND_OF[jnp.clip(code, 0, len(Items) - 1)]


def color_of(code):
    return COLOR_OF[jnp.clip(code, 0, len(Items) - 1)]


# Backwards-compatible names used throughout the step function, the tests
# and the analysis scripts: the placed waste codes / tool codes and their
# counts. "type k" == waste colour k; "tool t" == tool colour t.
DIRT_TYPES = ITEM_CODE[Kinds.dirt, :NUM_DIRT_COLORS]
NUM_DIRT_TYPES = NUM_DIRT_COLORS
TOOL_TYPES = ITEM_CODE[Kinds.tool, :NUM_TOOL_COLORS]
NUM_TOOLS = NUM_TOOL_COLORS
# Attribute chain variant (attr_rules): the NUM_DIRT_COLORS colours read as
# NUM_HUES hues x NUM_SHADES shades, colour = hue * NUM_SHADES + shade with
# shade 0 light, 1 mid, 2 dark. The palette below follows the same layout.
NUM_SHADES = 3
NUM_HUES = NUM_DIRT_COLORS // NUM_SHADES
assert NUM_HUES * NUM_SHADES == NUM_DIRT_COLORS


def is_dirt(x):
    """True where x is waste of any colour (clean water excluded)."""
    return (x >= DIRT_BASE) & (x < DIRT_BASE + NUM_DIRT_COLORS)


def is_tool(x):
    """True where x is a tool station of any colour (the default variant
    places NUM_TOOL_COLORS of them, the bijective variant NUM_DIRT_COLORS)."""
    return (x >= TOOL_BASE) & (x < TOOL_BASE + NUM_COLORS)


# Observation factorisation, XLand style: instead of one channel per item
# code, a cell shows WHICH KIND it is (one-hot over non-empty kinds) and
# WHICH COLOUR (one-hot over the shared colour axis, zero for uncoloured
# kinds). Built as linear maps from the flat per-code one-hot the pipeline
# already produces, so the rest of the observation code is untouched.
_kind_map = onp.zeros((len(Items) - 1, NUM_KINDS - 1), dtype=onp.int8)
_color_map = onp.zeros((len(Items) - 1, NUM_COLORS), dtype=onp.int8)
for _code in range(1, len(Items)):
    _kind_map[_code - 1, int(_kind_of[_code]) - 1] = 1
    if _kind_of[_code] in (Kinds.dirt, Kinds.tool):
        _color_map[_code - 1, int(_color_of[_code])] = 1
KIND_MAP = jnp.asarray(_kind_map)
COLOR_MAP = jnp.asarray(_color_map)
NUM_ITEM_CHANNELS = (NUM_KINDS - 1) + NUM_COLORS


# ---------------------------------------------------------------------------
# The hidden rule, XLand style: a rule is DATA, not code. A ruleset is a
# (MAX_RULES, RULE_ENCODING_LEN) uint8 array of [rule_id, args...] rows —
# see opensocialjax/environments/discovery_rules.py — carried in State and
# interpreted by the step function at runtime, so sampling and evaluation
# stay jit- and vmap-friendly and every vmapped environment can hold a
# different task.
#
# The vanilla task is the tool-compatibility table: waste type k is cleared
# only by tool table[k], encoded as six ToolClearsRule rows (plus EmptyRule
# padding). Firing the beam while holding any other tool does nothing at
# all. That "nothing at all" is the point — an earlier version made the wrong
# choice merely *slower*, and cleaning everything indiscriminately then scored
# within a few percent of an oracle, so no inference was ever needed.
#
# This mirrors XLand's AgentHoldRule: hold the right object and the production
# fires, hold the wrong one and nothing happens.
# ---------------------------------------------------------------------------
RULE_LEN = NUM_DIRT_TYPES


ENUMERABLE_LIMIT = 1_000_000


def rule_space_size():
    return NUM_TOOLS ** NUM_DIRT_TYPES


def all_rules():
    """Every dense tool table, NUM_TOOLS**NUM_DIRT_TYPES of them, generated
    vectorised (base-NUM_TOOLS digits of an index) so no Python list of
    tuples is ever built. For spaces beyond ENUMERABLE_LIMIT callers should
    sample instead (see rule_benchmark.build_benchmark) — 3**15 tables is
    ~430 MB here and ~1 GB encoded, which is fine on a GPU node but not on
    a login node."""
    n = rule_space_size()
    idx = onp.arange(n, dtype=onp.int64)
    digits = onp.empty((n, NUM_DIRT_TYPES), dtype=onp.int16)
    for k in range(NUM_DIRT_TYPES - 1, -1, -1):
        digits[:, k] = idx % NUM_TOOLS
        idx //= NUM_TOOLS
    return jnp.asarray(digits)


def sample_rule(key):
    """Draw a dense tool table uniformly: one tool per waste type."""
    return jax.random.randint(key, (RULE_LEN,), 0, NUM_TOOLS).astype(jnp.int16)


def all_rulesets():
    """The whole space as padded ruleset encodings, (14348907, MAX_RULES,
    RULE_ENCODING_LEN) — what `enumerate_benchmark` and the train/test split
    operate on."""
    return jax.vmap(encode_tool_table)(all_rules())


def sample_ruleset(key):
    """One ruleset encoding, uniform over the space — the fallback when the
    environment is built without a rule_pool."""
    return encode_tool_table(sample_rule(key))


def sample_chain_ruleset(key, depth=2, p_chain=0.5):
    """XLand-style production-chain ruleset, sampled in pure jax (vmappable).

    Every waste colour is either DIRECT (one ToolClearsRule: tool -> water)
    or, with probability ``p_chain`` when ``depth >= 2``, a two-link CHAIN:
    a ToolTransformsRule turns it into some *direct* colour, whose own rule
    then clears it. Chains therefore always terminate in exactly two hits
    and there are no dead ends; at least one colour is forced direct so a
    chain always has somewhere to land. ``depth=1`` gives the plain
    tool-table space (identical to ``sample_ruleset``).
    """
    k_tool, k_chain, k_anchor, k_next = jax.random.split(key, 4)
    n = NUM_DIRT_COLORS
    tools = jax.random.randint(k_tool, (n,), 0, NUM_TOOLS)
    is_chain = jax.random.uniform(k_chain, (n,)) < (p_chain if depth >= 2 else 0.0)
    anchor = jax.random.randint(k_anchor, (), 0, n)
    is_chain = is_chain.at[anchor].set(False)          # >= 1 direct colour
    # next colour for chain rows: uniform over the direct colours
    logits = jnp.where(is_chain, -jnp.inf, 0.0)
    nxt = jax.random.categorical(k_next, logits, shape=(n,))
    k = jnp.arange(n, dtype=jnp.uint8)
    rows = jnp.stack([
        jnp.where(is_chain, TOOL_TRANSFORMS, TOOL_CLEARS).astype(jnp.uint8),
        k,
        tools.astype(jnp.uint8),
        jnp.where(is_chain, nxt, 0).astype(jnp.uint8),
    ], axis=-1)
    return pad_ruleset(rows, MAX_RULES)


def sample_bijective_ruleset(key):
    """One-to-one rules: a random PERMUTATION of colours onto tools, so every
    waste colour has its own dedicated tool and every tool clears exactly
    one colour (NUM_DIRT_COLORS tools in play). A random tool then clears
    1/NUM_DIRT_COLORS of the river instead of 1/3 — blind and cycling
    strategies collapse and the rule is worth ~10x more. Space: 10! ~ 3.6M;
    benchmarks sample it."""
    return encode_tool_table(jax.random.permutation(key, NUM_DIRT_COLORS))


def rule_table(state):
    """Dense view of a state's hidden ruleset: table[k] = the tool the beam
    must carry to ACT on waste colour k (clear it, or transform it when the
    colour is a chain link; -1 if none). For diagnostics and scripted
    oracles only — the step function itself interprets the encoding."""
    return decode_chain(state.rule_encoding, NUM_DIRT_TYPES, NUM_SHADES)[0]


def craft_sequence(state, k=3):
    """The crafting variant's hidden pickup order, (k,) int16 (-1s if the
    ruleset is not a crafting ruleset)."""
    return decode_craft_sequence(state.rule_encoding, k)


def rule_next(state):
    """Per colour: what the beam turns it into — -1 for water, otherwise the
    next colour of the chain. All -1 for a plain tool-table ruleset."""
    return decode_chain(state.rule_encoding, NUM_DIRT_TYPES, NUM_SHADES)[1]


def attr_tables(state):
    """Attribute variant: the two hidden tables — ``hue_tools[h]`` clears
    light waste of hue h, ``shade_tools[s]`` lightens shade s (``-1`` at
    s = 0: light waste is cleared, never lightened)."""
    return decode_attr_tables(state.rule_encoding, NUM_HUES, NUM_SHADES)


def craft_recipes(state, k=2):
    """Attribute-craft variant: ``(products, recipes, valid)`` over the rule
    rows — ``recipes[i]`` is the station order that crafts tool
    ``products[i]`` where ``valid[i]``. Products are ``num_tools + j`` with
    j = hue 0..NUM_HUES-1, then mid, then dark."""
    valid, products, seqs = decode_craft_recipes(state.rule_encoding, k)
    return products, seqs, valid


def trick_tools(state):
    """Attribute variant: the tools this trial's dominant hue needs, in the
    order a dark cell needs them reversed — ``[T_h[dom hue], T_s[mid],
    T_s[dark]]``. The analogue of ``rule_table(state)[state.dom_type]``."""
    hue_tools, shade_tools = attr_tables(state)
    return jnp.concatenate([hue_tools[state.dom_type][None], shade_tools[1:]])


ROTATIONS = jnp.array(
    [
        [0, 0, 1],  # turn left
        [0, 0, -1],  # turn right
        [0, 0, 0],  # left
        [0, 0, 0],  # right
        [0, 0, 0],  # up
        [0, 0, 0],  # down
        [0, 0, 0],  # stay
        [0, 0, 0],  # zap
        [0, 0, 0],  # zap_clean
        [0, 0, 0],  # pick_up
    ],
    dtype=jnp.int8,
)

STEP = jnp.array(
    [
        [1, 0, 0],  # up
        [0, 1, 0],  # right
        [-1, 0, 0],  # down
        [0, -1, 0],  # left
    ],
    dtype=jnp.int8,
)

STEP_MOVE = jnp.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],  
        [0, -1, 0],  
        [1, 0, 0],  
        [-1, 0, 0],  
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],  # zap_clean
        [0, 0, 0],  # pick_up
    ],
    dtype=jnp.int8,
)

char_to_int = {
    'W': 1,
    ' ': 0,  # empty 0
    'A': 3,  # exist apple, not used in this environment
    'P': 4,  # spawn_point
    'Q': 5,  # spawn_point defence, not used in this environment
    'B': 6, # potential_apple
    'S': 7, # river
    'H': 8, # potential_dirt
    'F': 9, # actual_dirt
    '+': 0, # should be "sand", "shadow_e", "shadow_n"
    'f': 0, # should be "sand", "shadow_e", "shadow_n"
    ";": 0,
    ",": 0,
    "^": 0,
    "=": 0,
    ">": 0,
    "<": 0,
    "~": 7,
    "T": 6,
    

}


def ascii_map_to_matrix(map_ASCII, char_to_int):
    """
    Convert ASCII map to a JAX numpy matrix using the given character mapping.
    
    Args:
    map_ASCII (list): List of strings representing the ASCII map
    char_to_int (dict): Dictionary mapping characters to integer values
    
    Returns:
    jax.numpy.ndarray: 2D matrix representation of the ASCII map
    """
    # Determine matrix dimensions
    height = len(map_ASCII)
    width = max(len(row) for row in map_ASCII)
    
    # Create matrix filled with zeros
    matrix = jnp.zeros((height, width), dtype=jnp.int32)
    
    # Fill matrix with mapped values
    for i, row in enumerate(map_ASCII):
        for j, char in enumerate(row):
            matrix = matrix.at[i, j].set(char_to_int.get(char, 0))
    
    return matrix

# High-contrast, colourblind-friendly categorical palette: distinct on the
# sand, water and waste backgrounds (the old HSV wheel produced low-contrast
# near-duplicate colours at 6+ agents).
AGENT_PALETTE = [
    (230, 25, 75),    # crimson
    (245, 130, 48),   # orange
    (255, 225, 25),   # yellow
    (240, 50, 230),   # magenta
    (70, 240, 240),   # cyan
    (145, 30, 180),   # purple
    (60, 180, 75),    # green
    (0, 130, 200),    # azure
    (250, 190, 212),  # pink
    (170, 110, 40),   # brown
]


def generate_agent_colors(num_agents):
    return [AGENT_PALETTE[i % len(AGENT_PALETTE)] for i in range(num_agents)]


# ---- tile style helpers (rendering only, cached per tile) -----------------
SAND_COLOUR = (201, 192, 165)
AGENT_OUTLINE_COLOUR = (25, 25, 28)



def paint_wall_tile(img):
    img[:, :] = (99, 102, 106)
    fill_coords(img, point_in_rect(0.0, 1.0, 0.0, 0.08), (122, 125, 130))
    fill_coords(img, point_in_rect(0.0, 0.08, 0.0, 1.0), (122, 125, 130))
    fill_coords(img, point_in_rect(0.0, 1.0, 0.92, 1.0), (80, 82, 86))


SOIL, APPLE_ON_SOIL = 102, 103      # render-only codes: orchard ground, and an apple standing on it


def paint_soil_tile(img):
    """Orchard ground (random layouts): earth with furrows, so the orchard reads as one patch whether a cell is
    empty, carries an apple or has an agent on it."""
    img[:, :] = (176, 150, 112)
    for y0 in (0.18, 0.43, 0.68):
        fill_coords(img, point_in_rect(0.12, 0.88, y0, y0 + 0.07), (150, 124, 90))


def paint_apple_tile(img):
    fill_coords(img, point_in_circle(0.5, 0.54, 0.33), (150, 30, 30))
    fill_coords(img, point_in_circle(0.5, 0.54, 0.28), (214, 39, 40))
    fill_coords(img, point_in_rect(0.47, 0.53, 0.14, 0.30), (70, 140, 60))
    fill_coords(img, point_in_circle(0.41, 0.45, 0.06), (245, 200, 200))


def paint_beam_tile(img, glow, core, orient=None):
    """Beam segment: bright core + soft glow, oriented along the beam.

    orient: "h"/"v" for a segment whose neighbours continue horizontally/
    vertically, None for an isolated or corner cell (symmetric burst)."""
    if orient == "h":
        fill_coords(img, point_in_rect(0.0, 1.0, 0.18, 0.82), glow)
        fill_coords(img, point_in_rect(0.0, 1.0, 0.38, 0.62), core)
    elif orient == "v":
        fill_coords(img, point_in_rect(0.18, 0.82, 0.0, 1.0), glow)
        fill_coords(img, point_in_rect(0.38, 0.62, 0.0, 1.0), core)
    else:
        fill_coords(img, point_in_rect(0.15, 0.85, 0.15, 0.85), glow)
        fill_coords(img, point_in_rect(0.32, 0.68, 0.32, 0.68), core)


def paint_water_tile(img):
    img[:, :] = (48, 110, 212)
    fill_coords(img, point_in_line(0.08, 0.30, 0.45, 0.22, 0.035), (105, 158, 235))
    fill_coords(img, point_in_line(0.55, 0.62, 0.92, 0.55, 0.035), (105, 158, 235))
    fill_coords(img, point_in_line(0.25, 0.80, 0.60, 0.85, 0.030), (105, 158, 235))


# One palette per waste COLOUR, indexed by colour, generated as NUM_HUES hue
# families x NUM_SHADES lightness levels (colour = hue*NUM_SHADES + shade) so
# the attribute variant's two axes read on screen: which hue a cell is, and
# how dark. All variants share the palette; the plain table task's rule is
# index-agnostic, so the ordering hints at nothing there. Hue initials
# (R/Y/G/C/P) double as text glyphs and must not collide with T (tool).
ATTR_HUE_NAMES = ("red", "yellow", "green", "cyan", "purple")
ATTR_SHADE_NAMES = ("light", "mid", "dark")
_ATTR_HUE_DEG = (5, 50, 120, 180, 285)
_ATTR_SHADE_LIGHTNESS = (0.62, 0.42, 0.24)   # light, mid, dark
assert len(ATTR_HUE_NAMES) == len(_ATTR_HUE_DEG) == NUM_HUES
assert len(ATTR_SHADE_NAMES) == len(_ATTR_SHADE_LIGHTNESS) == NUM_SHADES


def _dirt_palette(hue_deg, lightness, saturation=0.38):
    """(base, blob, speck) for one waste colour: muted sludge of one hue at
    one lightness, blobs darker and specks lighter than the base."""
    def rgb(l):
        return tuple(int(round(255 * x))
                     for x in colorsys.hls_to_rgb(hue_deg / 360.0, l, saturation))
    return rgb(lightness), rgb(lightness * 0.72), rgb(min(0.92, lightness * 1.35))


DIRT_COLOUR_PALETTES = tuple(
    _dirt_palette(h, l) for h in _ATTR_HUE_DEG for l in _ATTR_SHADE_LIGHTNESS)
assert len(DIRT_COLOUR_PALETTES) >= NUM_DIRT_COLORS

# Tool station colours, indexed by tool colour. They are the ones a policy
# has to associate with a waste colour, so they are strong and unmistakably
# non-waste — a station must never be confused for something to clean.
TOOL_COLOURS = (
    (216, 92, 68),     # 0 rake — red
    (232, 176, 56),    # 1 skimmer — amber
    (84, 156, 216),    # 2 siphon — blue
    (120, 200, 120),   # 3 scoop — green
    (200, 120, 200),   # 4 (unplaced) magenta
    (240, 240, 120),   # 5 (unplaced) yellow
    (120, 220, 220),   # 6 (unplaced) cyan
    (230, 230, 230),   # 7 (unplaced) white
    (255, 140, 30),    # 8 (bijective variant) orange
    (170, 150, 255),   # 9 (bijective variant) lavender
    (255, 200, 200),   # 10 (unplaced)
    (200, 255, 200),   # 11 (unplaced)
    (200, 200, 255),   # 12 (unplaced)
    (255, 255, 200),   # 13 (unplaced)
    (255, 200, 255),   # 14 (unplaced)
)
assert len(TOOL_COLOURS) >= NUM_COLORS


def paint_tool_tile(img, kind):
    colour = TOOL_COLOURS[int(color_of(kind))]
    dark = tuple(int(c * 0.55) for c in colour)
    light = tuple(min(255, int(c * 1.25)) for c in colour)
    img[:, :] = (168, 160, 138)   # a paved plot, distinct from bare sand
    # a plinth with the tool's emblem on it, so the type reads at a glance
    fill_coords(img, point_in_rect(0.14, 0.86, 0.62, 0.86), dark)
    fill_coords(img, point_in_rect(0.18, 0.82, 0.58, 0.66), colour)
    fill_coords(img, point_in_circle(0.5, 0.38, 0.22), colour)
    fill_coords(img, point_in_circle(0.5, 0.38, 0.12), light)


def paint_dirt_tile(img, kind=Items.dirt_a):
    base, blob, speck = DIRT_COLOUR_PALETTES[int(color_of(kind))]
    img[:, :] = base
    for (cx, cy, r) in ((0.30, 0.30, 0.16), (0.72, 0.55, 0.14), (0.40, 0.78, 0.11)):
        fill_coords(img, point_in_circle(cx, cy, r), blob)
    for (cx, cy) in ((0.60, 0.22), (0.18, 0.60), (0.85, 0.85)):
        fill_coords(img, point_in_circle(cx, cy, 0.045), speck)


AGENT_SKIN_COLOUR = (240, 200, 165)
AGENT_INK_COLOUR = (30, 28, 32)


def _agent_sprite(agent_dir, color):
    """Pixel-art humanoid on a 32x32 grid -> (rgb, alpha).

    dir 0 faces the viewer (down), 1 = right profile, 2 = back view (up),
    3 = left profile, matching the direction the agent's beam fires.
    Side views are true profiles: narrow torso, one visible arm, striding
    legs and a nose bump."""
    g = 32
    rgb = onp.zeros((g, g, 3), dtype=onp.uint8)
    alpha = onp.zeros((g, g), dtype=bool)
    hair = tuple(int(c * 0.45) for c in color)
    pants = tuple(int(c * 0.62) for c in color)
    skin = AGENT_SKIN_COLOUR
    ink = AGENT_INK_COLOUR
    mouth = tuple(int(c * 0.8) for c in skin)

    def rect(x0, y0, x1, y1, col):
        rgb[y0:y1, x0:x1] = col
        alpha[y0:y1, x0:x1] = True

    if agent_dir in (0, 2):                       # front / back view
        rect(12, 22, 15, 27, pants)               # legs
        rect(17, 22, 20, 27, pants)
        rect(11, 27, 15, 29, ink)                 # feet
        rect(17, 27, 21, 29, ink)
        rect(9, 15, 12, 21, color)                # arms
        rect(20, 15, 23, 21, color)
        rect(9, 21, 12, 23, skin)                 # hands
        rect(20, 21, 23, 23, skin)
        rect(11, 14, 21, 22, color)               # torso
        rect(11, 5, 21, 13, skin)                 # head
        rect(10, 6, 11, 12, hair)
        rect(21, 6, 22, 12, hair)
        rect(11, 4, 21, 8, hair)                  # hair cap
        rect(10, 5, 11, 7, hair)
        rect(21, 5, 22, 7, hair)
        if agent_dir == 0:
            rect(13, 9, 15, 11, ink)              # eyes
            rect(17, 9, 19, 11, ink)
            rect(15, 12, 17, 13, mouth)
        else:
            rect(11, 4, 21, 13, hair)             # back of head
            rect(10, 5, 22, 12, hair)
    else:                                         # side profile
        rect(17, 22, 20, 26, pants)               # front leg
        rect(12, 22, 15, 27, pants)               # back leg
        rect(18, 26, 23, 28, ink)                 # front foot
        rect(10, 27, 15, 29, ink)                 # back foot
        rect(13, 14, 20, 22, color)               # narrow torso
        rect(17, 15, 20, 21, tuple(int(c * 0.78) for c in color))   # arm
        rect(18, 21, 21, 23, skin)                # hand
        rect(12, 5, 21, 13, skin)                 # head
        rect(21, 9, 22, 11, skin)                 # nose
        rect(12, 4, 21, 8, hair)                  # hair cap
        rect(11, 5, 13, 13, hair)                 # back hair
        rect(11, 12, 14, 14, hair)                # nape
        rect(17, 9, 19, 11, ink)                  # single eye
        rect(19, 12, 21, 13, mouth)

    if agent_dir == 3:                            # mirror for left facing
        rgb = rgb[:, ::-1].copy()
        alpha = alpha[:, ::-1].copy()
    return rgb, alpha


def draw_agent(img, agent_dir, color):
    """Blit the humanoid sprite (with a 1px outline) onto a tile image."""
    rgb, alpha = _agent_sprite(agent_dir, color)
    grown = alpha.copy()
    grown[1:, :] |= alpha[:-1, :]
    grown[:-1, :] |= alpha[1:, :]
    grown[:, 1:] |= alpha[:, :-1]
    grown[:, :-1] |= alpha[:, 1:]
    rgb[grown & ~alpha] = AGENT_OUTLINE_COLOUR

    scale = img.shape[0] // 32
    mask = onp.repeat(onp.repeat(grown, scale, 0), scale, 1)
    sprite = onp.repeat(onp.repeat(rgb, scale, 0), scale, 1)
    img[mask] = sprite[mask]

GREEN_COLOUR = (44.0, 160.0, 44.0)
RED_COLOUR = (214.0, 39.0, 40.0)
###################################################

class OpenCleanup(MultiAgentEnv):

    # used for caching
    tile_cache: Dict[Tuple[Any, ...], Any]

    def __init__(
        self,
        # A meta-episode is num_outer_steps trials of num_inner_steps steps.
        # The hidden rule is fixed for the whole meta-episode, so an agent can
        # spend early trials working it out and later trials exploiting it —
        # the structure RL^2-style recurrent policies need.
        num_inner_steps=1000,
        num_outer_steps=10,
        num_agents=7,
        inequity_aversion=False,
        inequity_aversion_target_agents=None,
        inequity_aversion_alpha=5,
        inequity_aversion_beta=0.05,
        svo=False,
        svo_target_agents=None,
        svo_w=0.5,
        svo_ideal_angle_degrees=45,
        enable_smooth_rewards=False,
        interest=False,
        s_interest = 0.5,
        s_interest_schedule=None,
        s_interest_change_every=30000000,
        cf=False,
        cf_alpha=1,
        maxAppleGrowthRate=0.05, 
        thresholdDepletion=0.4,  # 0.4
        # Optional pool of rules this instance may draw from (a
        # RuleBenchmark). Left None the environment samples the whole rule
        # space as before; passing the train or test half of a benchmark is
        # what makes held-out evaluation possible.
        rule_pool=None,
        # Oracle-input control: when True, the hidden rule is appended to the
        # observation as NUM_DIRT_TYPES x NUM_TOOLS one-hot channels, so a
        # memoryless policy can be trained WITH the rule. This bounds what
        # "knowing the rule" is worth to a *trained* policy — if oracle-fed
        # IPPO also lands far below the scripted oracle, the gap RPPO failed
        # to close was never inference, and no memory fix can close it.
        # Never set this in a run whose score is meant to require inference.
        reveal_rule=False,
        # Per-trial waste skew: when on, each trial draws one dominant waste
        # type and `waste_skew` of the waste (initial cells and respawns) is
        # that type. This raises the value of the FULL rule table: within a
        # trial only 1-2 mappings matter and they change at every trial
        # boundary, so a policy that carries the table across trials opens
        # each trial with the right tool while a within-trial prober has to
        # rediscover it — the pressure the uniform mix never applied (a
        # prober knowing 1.5/6 mappings already saturated throughput).
        skewed_waste=False,
        waste_skew=0.8,
        # one dominant waste hue for the WHOLE meta-episode instead of a fresh
        # draw every trial: the trial chain then re-tests the SAME three tools
        # (dark, mid, the hue's) rather than a new hue's each trial
        dom_hue_per_episode=False,
        # Pin the dominant waste hue (or type) instead of drawing it: with a one-map layout_pool and a one-rule
        # rule_pool it replays the same problem under other randomness (spawns, waste shades, respawns). None = draw.
        fixed_dom_type=None,
        # Cleaning chains (XLand production rules). chain_depth=1 is the
        # plain tool table. chain_depth=2: each waste colour is, with
        # probability p_chain, a two-hit chain — its tool turns it into
        # another (direct) colour instead of clearing it. Only consulted
        # when no rule_pool is given; benchmarks carry their own rulesets.
        chain_depth=1,
        p_chain=0.5,
        # One-to-one rules (see sample_bijective_ruleset): tools = colours,
        # rules are permutations. Raises the value of knowing the rule ~10x
        # by making any fixed tool nearly useless — deliberately HARDER than
        # the default; the default stays the 3-tool table.
        bijective_rules=False,
        # Crafting rules. The hidden rule is an ORDER of station pickups
        # (e.g. tool 0, then 2, then 1). Walking onto stations in that order
        # — judged on the last craft_len pickups, a sliding window, with no
        # feedback in between — hands the agent THE cleaning tool: from then
        # on its beam clears waste of every colour and stations are inert.
        # The tool is lost at every trial boundary (the map is re-laid) while
        # the order stays fixed for the meta-episode, so remembering the
        # order is exactly what a later trial pays for. craft_tools tool
        # colours are placed (repeats allowed in the order), so the rule
        # space is craft_tools ** craft_len (64 by default).
        craft_rules=False,
        craft_tools=4,
        craft_len=3,
        # Whether the observation tells the agent that it holds the crafted
        # tool. Default False: the only way to find out whether the last
        # pickups were the right order is to FIRE at waste and see whether
        # it clears — the feedback comes from the river, not from a flag.
        craft_show_tool=False,
        # Explicit pickup (2026-09-03): walking onto or across a station does
        # NOTHING to the hand; the agent must stand on a station and take the
        # pick_up action (Actions.pick_up = 9, which only exists in the action
        # space when this is on). Removes the "lost my tool by walking past a
        # station" failure that dominated the LLM probes. Default off: RL
        # checkpoints and every earlier result use walk-on pickup.
        explicit_pickup=False,
        # Attribute chain (2026-08-30). Colours read as NUM_HUES hues x
        # NUM_SHADES shades (colour = hue*3 + shade; 0 light, 1 mid, 2 dark)
        # and the hidden rule is two small tables instead of one entry per
        # colour: T_h[hue] -> the tool that clears LIGHT waste of that hue to
        # water; T_s[shade] -> the tool that turns MID / DARK waste one shade
        # lighter (same hue), shared by every hue. A dark cell is three shots
        # with up to three tools, a light one a single shot; every shot's
        # effect is immediate (cleared / lighter / nothing). Knowing one
        # colour's tools therefore says something about the other fourteen —
        # the compressibility the independent-entry tables lacked. Sampling
        # only forces the two shade tools apart (T_s[mid] != T_s[dark]); with
        # attr_tools=4 that is P(4,2) * 4**5 = 12288 rulesets. Per-trial skew
        # is on the HUE: `waste_skew` of the waste is the dominant hue in all
        # three shades, so one tool clears at most a third of the river.
        attr_rules=False,
        attr_tools=4,
        # per_colour_tools (with attr_rules + craft_rules, 2026-09-16): every one of the 15 waste colours has its
        # OWN working tool and its own hidden recipe -- the tool for a light colour clears it, the tool for a mid or
        # dark colour turns it one shade lighter, and it does nothing to any other colour. 15 recipes instead of 7:
        # P(craft_tools**craft_len, 15) rulesets (2.1e13 at 4 tools x 2 pick-ups), and nothing carries over between
        # hues. Crafted tool ids are TYPE_TOOL_BASE + colour (hand-only, never placed on the grid).
        per_colour_tools=False,
        # attr_rules + craft_rules together (2026-08-30) = the attribute-CRAFT
        # variant: every acting tool is crafted. The base tools on the stations
        # do nothing to waste by themselves; entering stations in a hidden
        # order (the last craft_len pickups, a sliding window) puts a crafted
        # tool in the agent's ONE hand — one per hue (clears that hue's light
        # waste) and one per shade (mid, dark: lightens it) — and the next
        # station entered replaces it with that base tool again. Nothing
        # signals a craft; only a shot's effect does. Recipes are distinct
        # orders drawn per meta-episode (P(craft_tools**craft_len, 7) =
        # 57,657,600 at 4 tools x 2 pickups), lost with the tool at every
        # trial boundary. craft_len must be 2 here (a recipe row holds
        # [id, product, t0, t1]).
        # Row the tool stations are laid out along; defaults to
        # the open ground between the river and the orchard.
        tool_row=None,
        # Random maps (2026-09-17): a layouts.LayoutPool. One layout is drawn per meta-episode and kept across its
        # trials; river, stream, orchard, stations and spawn cells come from it instead of map_ASCII (which then only
        # fixes the grid size). None = the fixed map, bit-identical to before.
        layout_pool=None,
        # True: info also carries the rule's verdict on every beam target -- beam_rc (agent, 4, 2),
        # beam_hit / beam_before / beam_after (agent, 4). Off by default because the RL loops reshape each
        # info leaf to one value per actor.
        report_beam=False,
        thresholdRestoration=0.0,
        # Vanilla Clean Up's spawn rate, restored 2026-08-29. From 2026-08-13
        # to 2026-08-29 this defaulted to 0.1 — every run recorded in the
        # README up to seed 962 trained and was evaluated at 0.1 — because
        # tool-gating cuts a random agent's effective cleaning rate to
        # 1/NUM_TOOLS and at 0.5 the river settles far above the pollution
        # threshold (design gate: first reward 63 cells of net cleaning
        # away vs 23 at 0.1; tests/sweep_spawn_rate.py). Pass 0.1 explicitly
        # to reproduce those runs or to evaluate their checkpoints.
        dirtSpawnProbability=0.5,
        # Fraction of the map's waste cells that actually start dirty. At 1.0
        # every trial opens above the pollution threshold, so no reward exists
        # until the group has cleared its way down to it — and with cleaning
        # gated on the right tool that wall is tall enough that a learner sees
        # identically zero for an entire run.
        #
        # Pass a (lo, hi) pair instead and the fraction is drawn per trial.
        # That is the training curriculum, and it is randomised rather than
        # annealed on a schedule: a batch then always contains trials that
        # open below the threshold (reward from step one, so there is always a
        # gradient) alongside trials that open above it (where choosing the
        # right tool is what decides the return). A staged schedule would need
        # its switch point tuned, would risk forgetting the easy regime at the
        # switch, and would have to split the training loop into separately
        # jitted phases; a range needs none of that.
        #
        # Evaluation should stay pinned at 1.0 — the regime the design gate
        # was measured in.
        initial_dirt=1.0,
        delayStartOfDirtSpawning=50, # 50
        jit=True,
        
        obs_size=11,
        cnn=True,

        map_ASCII = [
                'HFFFHFFHFHFHFHFHFHFHHFHFFFHF',
                'HFHFHFFHFHFHFHFHFHFHHFHFFFHF',
                'HFFHFFHHFHFHFHFHFHFHHFHFFFHF',
                'HFHFHFFHFHFHFHFHFHFHHFHFFFHF',
                'HFFFFFFHFHFHFHFHFHFHHFHFFFHF',
                '==============+~FHHHHHHf====',
                '   P    P      ===+~SSf     ',
                '     P     P   P  <~Sf  P   ',
                '             P   P<~S>      ',
                '   P    P         <~S>   P  ',
                '               P  <~S>P     ',
                '     P           P<~S>      ',
                '           P      <~S> P    ',
                '  P             P <~S>      ',
                '^T^T^T^T^T^T^T^T^T;~S,^T^T^T',
                'BBBBBBBBBBBBBBBBBBBssBBBBBBB',
                'BBBBBBBBBBBBBBBBBBBBBBBBBBBB',
                'BBBBBBBBBBBBBBBBBBBBBBBBBBBB',
                'BBBBBBBBBBBBBBBBBBBBBBBBBBBB',
            ]
    ):

        super().__init__(num_agents=num_agents)

        self.maxAppleGrowthRate = maxAppleGrowthRate
        self.thresholdDepletion = thresholdDepletion
        self.rule_pool = rule_pool
        self.reveal_rule = reveal_rule
        self.skewed_waste = skewed_waste
        self.waste_skew = waste_skew
        self.dom_hue_per_episode = bool(dom_hue_per_episode)
        self.fixed_dom_type = None if fixed_dom_type is None else int(fixed_dom_type)
        self.chain_depth = chain_depth
        self.p_chain = p_chain
        self.bijective_rules = bijective_rules
        self.craft_rules = bool(craft_rules)
        self.explicit_pickup = bool(explicit_pickup)
        self.craft_tools = int(craft_tools)
        self.craft_len = int(craft_len)
        self.craft_show_tool = bool(craft_show_tool)
        if self.craft_rules:
            assert self.craft_len <= RULE_ENCODING_LEN - 1, "craft_len must fit one rule row"
            assert 1 + self.craft_tools * self.craft_len <= NUM_COLORS, (
                "crafted flag + pickup one-hots must fit the held-tool channels")
            assert not reveal_rule, "reveal_rule is not implemented for craft_rules"
        self.attr_rules = bool(attr_rules)
        self.attr_tools = int(attr_tools)
        self.attr_craft = self.attr_rules and self.craft_rules
        self.per_colour_tools = bool(per_colour_tools)
        if self.per_colour_tools:
            assert self.attr_craft, "per_colour_tools is a mode of the attribute-craft variant (attr_rules + craft_rules)"
        if self.attr_rules:
            assert not (bijective_rules or chain_depth >= 2), "attr_rules is its own variant"
            assert NUM_SHADES - 1 < self.attr_tools <= NUM_COLORS, \
                "attr_tools must exceed the number of shade tools"
            assert NUM_HUES + NUM_SHADES - 1 <= MAX_RULES
            assert not reveal_rule, "reveal_rule is not implemented for attr_rules"
        if self.attr_craft and self.per_colour_tools:
            n_products = NUM_DIRT_COLORS
            assert self.craft_len == RULE_ENCODING_LEN - 2, "per-colour recipes are two pickups"
            assert self.craft_tools ** self.craft_len >= n_products, "not enough pick-up orders for one recipe per colour"
            assert n_products <= MAX_RULES, "one recipe row per colour must fit"
            self.attr_tools = self.craft_tools
        elif self.attr_craft:
            n_products = NUM_HUES + NUM_SHADES - 1
            assert self.craft_len == RULE_ENCODING_LEN - 2, "attr-craft recipes are two pickups"
            assert self.craft_tools ** self.craft_len >= n_products, "not enough orders for distinct recipes"
            assert 2 * n_products <= MAX_RULES, "hue/shade rows + recipe rows must fit"
            assert self.craft_tools + n_products <= NUM_COLORS, "crafted tool ids must fit the tool axis"
            self.attr_tools = self.craft_tools
        # how many tool colours are placed / sampled in this variant
        self.num_tools = (self.craft_tools if self.craft_rules
                          else self.attr_tools if self.attr_rules
                          else NUM_DIRT_COLORS if bijective_rules else NUM_TOOL_COLORS)
        self.thresholdRestoration = thresholdRestoration
        self.dirtSpawnProbability = dirtSpawnProbability
        # scalar => fixed; (lo, hi) => drawn per trial. Duck-typed rather
        # than isinstance'd: Hydra hands this over as an omegaconf ListConfig,
        # which is neither a list nor a tuple.
        if hasattr(initial_dirt, "__len__") and not isinstance(initial_dirt, str):
            self.initial_dirt = None
            self.initial_dirt_range = (float(initial_dirt[0]),
                                       float(initial_dirt[1]))
        else:
            self.initial_dirt = float(initial_dirt)
            self.initial_dirt_range = None
        self.delayStartOfDirtSpawning = delayStartOfDirtSpawning
        self.inequity_aversion = inequity_aversion
        self.inequity_aversion_target_agents = inequity_aversion_target_agents
        self.inequity_aversion_alpha = inequity_aversion_alpha
        self.inequity_aversion_beta = inequity_aversion_beta
        self.svo = svo
        self.svo_target_agents = svo_target_agents
        self.svo_w = svo_w
        self.svo_ideal_angle_degrees = svo_ideal_angle_degrees
        self.smooth_rewards = enable_smooth_rewards
        self.interest = interest
        self.s_interest = s_interest
        # Convert schedule to JAX array for JIT compatibility
        if s_interest_schedule is not None:
            self.s_interest_schedule = jnp.array(s_interest_schedule)
        else:
            self.s_interest_schedule = None
        self.s_interest_change_every = s_interest_change_every
        self.cnn = cnn
        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps
        self.cf = cf
        self.cf_alpha = cf_alpha
        self.agents = list(range(num_agents))#, dtype=jnp.int16)
        self._agents = jnp.array(self.agents, dtype=jnp.int16) + len(Items)

        self.PLAYER_COLOURS = generate_agent_colors(num_agents)
        # per-instance tile cache: cached agent colours depend on num_agents
        self.tile_cache = {}
        self.GRID_SIZE_ROW = len(map_ASCII)
        self.GRID_SIZE_COL = len(map_ASCII[0])
        self.OBS_SIZE = obs_size
        self.PADDING = self.OBS_SIZE - 1

        GRID = jnp.zeros(
            (self.GRID_SIZE_ROW + 2 * self.PADDING, self.GRID_SIZE_COL + 2 * self.PADDING),
            dtype=jnp.int16,
        )

        # First layer of padding is Wall
        GRID = GRID.at[self.PADDING - 1, :].set(5)
        GRID = GRID.at[self.GRID_SIZE_ROW + self.PADDING, :].set(5)
        GRID = GRID.at[:, self.PADDING - 1].set(5)
        self.GRID = GRID.at[:, self.GRID_SIZE_COL + self.PADDING].set(5)

        def find_positions(grid_array, letter):
            a_positions = jnp.array(jnp.where(grid_array == letter)).T
            return a_positions

        nums_map = ascii_map_to_matrix(map_ASCII, char_to_int)
        self.POTENTIAL_APPLE = find_positions(nums_map, char_to_int['B'])

        self.SPAWNS_PLAYER_IN = find_positions(nums_map, char_to_int['Q'])
        self.SPAWNS_PLAYERS = find_positions(nums_map, char_to_int['P'])
        assert len(self.SPAWNS_PLAYERS) >= num_agents, "need at least num_agents spawn cells for occupancy-aware respawn"
        self.SPAWNS_WALL = find_positions(nums_map, char_to_int['W'])
        self.RIVER = find_positions(nums_map, char_to_int['S'])
        self.POTENTIAL_DIRT = find_positions(nums_map, char_to_int['H'])
        self.DIRT = find_positions(nums_map, char_to_int['F'])

        # Tool stations, laid out as a single row across the open ground
        # between the river and the orchard — one every other cell, so the
        # supply line reads at a glance and fetching a tool is the same kind
        # of trip wherever an agent happens to be. They are placed
        # programmatically rather than written into map_ASCII, so the map
        # stays byte-identical to clean_up's and the two environments remain
        # directly comparable.
        def _tool_cols(r):
            return [c for c in range(0, len(map_ASCII[r]), 2)
                    if map_ASCII[r][c] == ' ']

        if tool_row is not None:
            t_row = tool_row
        else:
            # Pick the row that fits the most stations. Choosing by hand
            # would strand a whole bank of tools on one side of the river the
            # first time the map changes; this keeps both banks supplied.
            t_row = max(range(len(map_ASCII)),
                        key=lambda r: (len(_tool_cols(r)), -r))
        t_cols = _tool_cols(t_row)
        # trim to a whole number of tools so none is over-represented
        # trim to a whole number of tools so none is over-represented (the
        # bijective variant has more tools than stations and reports that
        # itself below)
        _per = self.num_tools if self.num_tools <= len(t_cols) else NUM_TOOLS
        t_cols = t_cols[:len(t_cols) - len(t_cols) % _per]
        assert t_cols, f"row {t_row} has no room for a line of tool stations"
        self.TOOLS = jnp.array([(t_row, c) for c in t_cols], dtype=jnp.int16)
        # the row cycles through the tools, so neighbours always differ
        if self.num_tools > len(t_cols):
            raise ValueError(
                f"{self.num_tools} tool colours but only {len(t_cols)} stations on "
                f"this map — the bijective variant needs a station per tool")
        self._tool_ids = jnp.arange(len(t_cols), dtype=jnp.int16) % self.num_tools
        self._tool_labels = ITEM_CODE[Kinds.tool, self._tool_ids]
        # Lookup used by the step function: tool index at a cell, -1 elsewhere.
        tool_grid = jnp.full((self.GRID_SIZE_ROW, self.GRID_SIZE_COL), -1,
                             dtype=jnp.int16)
        self._tool_grid = tool_grid.at[self.TOOLS[:, 0],
                                       self.TOOLS[:, 1]].set(self._tool_ids)
        # static host-side river cells for rendering (terrain under agents)
        self._river_cells = set(map(tuple, onp.asarray(self.RIVER).tolist()))

        self.layout_pool = layout_pool
        self.report_beam = bool(report_beam)
        if layout_pool is not None:
            assert tuple(layout_pool.grid_shape) == (self.GRID_SIZE_ROW, self.GRID_SIZE_COL), (layout_pool.grid_shape, self.GRID_SIZE_ROW, self.GRID_SIZE_COL)
            self._LP = {k: jnp.asarray(v, dtype=jnp.int16) for k, v in layout_pool.arrays.items()}
            n_layouts, n_st = self._LP["tools"].shape[:2]
            assert n_st >= self.num_tools, f"{n_st} stations per layout, {self.num_tools} tools"
            # stations are stored in chain order, so the tools cycle along the chain and neighbours differ
            self._tool_ids = jnp.arange(n_st, dtype=jnp.int16) % self.num_tools
            self._tool_labels = ITEM_CODE[Kinds.tool, self._tool_ids]
            tg = jnp.full((n_layouts, self.GRID_SIZE_ROW, self.GRID_SIZE_COL), -1, dtype=jnp.int16)
            self._LP_tool_grid = tg.at[jnp.arange(n_layouts)[:, None], self._LP["tools"][:, :, 0], self._LP["tools"][:, :, 1]].set(
                jnp.broadcast_to(self._tool_ids, (n_layouts, n_st)))
            # layout 0 stands in for the static arrays (every layout has the same number of cells of each kind)
            self.RIVER, self.POTENTIAL_DIRT, self.DIRT = self._LP["river"][0], self._LP["potential_dirt"][0], self._LP["dirt"][0]
            self.POTENTIAL_APPLE, self.SPAWNS_PLAYERS, self.TOOLS = self._LP["apple"][0], self._LP["spawns"][0], self._LP["tools"][0]
            self.SPAWNS_PLAYER_IN = jnp.zeros((0, 2), dtype=jnp.int16)
            self.SPAWNS_WALL = jnp.zeros((0, 2), dtype=jnp.int16)
            self._tool_grid = self._LP_tool_grid[0]
            assert len(self.SPAWNS_PLAYERS) >= num_agents
        self._layout_host_cache = {}

        def _layout(layout_id):
            """The cell arrays of the map a state plays on (traced; static arrays without a pool)."""
            if self.layout_pool is None:
                return dict(river=self.RIVER, potential_dirt=self.POTENTIAL_DIRT, dirt=self.DIRT, apple=self.POTENTIAL_APPLE,
                            spawns=self.SPAWNS_PLAYERS, spawns_in=self.SPAWNS_PLAYER_IN, walls=self.SPAWNS_WALL,
                            tools=self.TOOLS, tool_grid=self._tool_grid)
            return dict(river=self._LP["river"][layout_id], potential_dirt=self._LP["potential_dirt"][layout_id],
                        dirt=self._LP["dirt"][layout_id], apple=self._LP["apple"][layout_id],
                        spawns=self._LP["spawns"][layout_id], spawns_in=self.SPAWNS_PLAYER_IN, walls=self.SPAWNS_WALL,
                        tools=self._LP["tools"][layout_id], tool_grid=self._LP_tool_grid[layout_id])
        self._layout = _layout


        
        
        # first attempt at func - needs improvement
        # inefficient due to double-checking collisions
        

        def to_dict(
                agent: int,
                obs: jnp.ndarray,
                agent_invs: jnp.ndarray,
                agent_pickups: jnp.ndarray,
                inv_to_show: jnp.ndarray
            ) -> dict:
            '''
            Function to produce observation/state dictionaries.
            
            Args:
                - agent: int, number identifying agent
                - obs: jnp.ndarray, the combined grid observations for each
                agent
                - agent_invs: jnp.ndarray of current agents' inventories
                - agent_pickups: boolean indicators of interaction
                - inv_to_show: jnp.ndarray inventory to show to other agents
                
            Returns:
                - dictionary of full state observation.
            '''
            idx = agent - len(Items)
            state_dict = {
                "observation": obs,
                "inventory": {
                    "agent_inv": agent_invs,
                    "agent_pickups": agent_pickups,
                    "invs_to_show": jnp.delete(
                        inv_to_show,
                        idx,
                        assume_unique_indices=True
                    )
                }
            }

            return state_dict
        
        def combine_channels(
                grid: jnp.ndarray,
                agent: int,
                angles: jnp.ndarray,
                agent_pickups: jnp.ndarray,
                state: State,
            ):

            def move_and_collapse(
                    x: jnp.ndarray,
                    angle: jnp.ndarray,
                ) -> jnp.ndarray:

                # get agent's one-hot
                agent_element = jnp.array([jnp.int8(x[agent])])

                # mask to check if any other agent exists there
                mask = x[len(Items)-1:] > 0

                # does an agent exist which is not the subject?
                other_agent = jnp.int8(
                    jnp.logical_and(
                        jnp.any(mask),
                        jnp.logical_not(
                            agent_element
                        )
                    )
                )

                # what is the class of the item in cell
                item_idx = jnp.where(
                    x,
                    size=1
                )[0]

                # check if agent is frozen and can observe inventories
                show_inv_bool = jnp.logical_and(
                        state.freeze[
                            agent-len(Items)
                        ].max(axis=-1) > 0,
                        item_idx >= len(Items)
                )

                show_inv_idxs = jnp.where(
                    state.freeze[agent],
                    size=12, # since, in a setting where simultaneous interac-
                    fill_value=-1 # -tions can happen, only a max of 12 can
                )[0] # happen at once (zap logic), regardless of pop size

                inv_to_show = jnp.where(
                    jnp.logical_or(
                        jnp.logical_and(
                            show_inv_bool,
                            jnp.isin(item_idx-len(Items), show_inv_idxs),
                        ),
                        agent_element
                    ),
                    state.agent_invs[item_idx - len(Items)],
                    jnp.array([0, 0], dtype=jnp.int8)
                )[0]

                # check if agent is not the subject & is frozen & therefore
                # not possible to interact with
                frozen = jnp.where(
                    other_agent,
                    state.freeze[
                        item_idx-len(Items)
                    ].max(axis=-1) > 0,
                    0
                )

                # get pickup/inv info
                pick_up_idx = jnp.where(
                    jnp.any(mask),
                    jnp.nonzero(mask, size=1)[0],
                    jnp.int8(-1)
                )
                picked_up = jnp.where(
                    pick_up_idx > -1,
                    agent_pickups[pick_up_idx],
                    jnp.int8(0)
                )

                # Which tool this agent is carrying, one-hot. An agent that
                # cannot tell what it is holding cannot apply the rule at all,
                # so this is not a hint — it is the difference between a hard
                # task and an unsolvable one. It rides along in the per-cell
                # feature vector exactly as `angle` and `picked_up` do.
                # one-hot over the shared COLOUR axis, so "the tool I hold"
                # and "the colour of the cell in front of me" index the
                # same channels
                if self.craft_rules:
                    # crafting variant: [have the tool?] + one-hot of each of
                    # the last craft_len pickups (-1 -> all zeros), zero-padded
                    # to the same NUM_COLORS-wide block
                    a_i = agent - len(Items)
                    flag = (state.crafted[a_i].astype(jnp.int8) if self.craft_show_tool
                            else jnp.int8(0))
                    held = jnp.concatenate([
                        flag[None],
                        jax.nn.one_hot(state.pickups[a_i], self.craft_tools,
                                       dtype=jnp.int8).reshape(-1),
                    ])
                    held = jnp.pad(held, (0, NUM_COLORS - held.shape[0]))
                else:
                    held = jax.nn.one_hot(
                        state.held_tool[agent - len(Items)], NUM_COLORS,
                        dtype=jnp.int8)

                # build extension
                parts = [
                    agent_element,
                    other_agent,
                    angle,
                    picked_up,
                    inv_to_show,
                    frozen,
                    held,
                ]
                if self.reveal_rule:
                    # Oracle-input control (see the constructor): the decoded
                    # rule table rides along in every cell's feature vector,
                    # one NUM_TOOLS-wide one-hot block per waste type. A
                    # Python-level flag, so the default observation is
                    # byte-for-byte what it always was.
                    parts.append(jax.nn.one_hot(
                        decode_tool_table(state.rule_encoding, NUM_DIRT_TYPES),
                        NUM_TOOLS, dtype=jnp.int8).reshape(-1))
                extension = jnp.concatenate(parts, axis=-1)

                # build final feature vector: (kind one-hot, colour one-hot)
                # instead of one channel per item code — the cell's two
                # XLand axes, read off the flat one-hot with fixed maps
                item = x[:len(Items)-1].astype(jnp.int32)
                kind_vec = (item @ KIND_MAP.astype(jnp.int32)).astype(jnp.int8)
                color_vec = (item @ COLOR_MAP.astype(jnp.int32)).astype(jnp.int8)
                final_vec = jnp.concatenate(
                    [kind_vec, color_vec, extension],
                    axis=-1
                )

                return final_vec

            new_grid = jax.vmap(
                jax.vmap(
                    move_and_collapse
                )
            )(grid, angles)
            return new_grid
        
        def check_relative_orientation(
                agent: int,
                agent_locs: jnp.ndarray,
                grid: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Check's relative orientations of all other agents in view of
            current agent.
            
            Args:
                - agent: int, an index indicating current agent number
                - agent_locs: jax ndarray of agent locations (x, y, direction)
                - grid: jax ndarray of current agent's obs grid
                
            Returns:
                - grid with 1) int -1 in places where no agent exists, or
                where the agent is the current agent, and 2) int in range
                0-3 in cells of opposing agents indicating relative
                orientation to current agent.
            '''
            # we decrement by num of Items when indexing as we incremented by
            # 5 in constructor call (due to 5 non-agent Items enum & locations
            # are indexed from 0)
            idx = agent - len(Items)
            agents = jnp.delete(
                self._agents,
                idx,
                assume_unique_indices=True
            )
            curr_agent_dir = agent_locs[idx, 2]

            def calc_relative_direction(cell):
                cell_agent = cell - len(Items)
                cell_direction = agent_locs[cell_agent, 2]
                return (cell_direction - curr_agent_dir) % 4

            angle = jnp.where(
                jnp.isin(grid, agents),
                jax.vmap(calc_relative_direction)(grid),
                -1
            )

            return angle
        
        def rotate_grid(agent_loc: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
            '''
            Rotates agent's observation grid k * 90 degrees, depending on agent's
            orientation.

            Args:
                - agent_loc: jax ndarray of agent's x, y, direction
                - grid: jax ndarray of agent's obs grid

            Returns:
                - jnp.ndarray of new rotated grid.

            '''
            grid = jnp.where(
                agent_loc[2] == 1,
                jnp.rot90(grid, k=1, axes=(0, 1)),
                grid,
            )
            grid = jnp.where(
                agent_loc[2] == 2,
                jnp.rot90(grid, k=2, axes=(0, 1)),
                grid,
            )
            grid = jnp.where(
                agent_loc[2] == 3,
                jnp.rot90(grid, k=3, axes=(0, 1)),
                grid,
            )

            return grid

        def _get_obs_point(agent_loc: jnp.ndarray) -> jnp.ndarray:
            '''
            Obtain the position of top-left corner of obs map using
            agent's current location & orientation.

            Args: 
                - agent_loc: jnp.ndarray, agent x, y, direction.
            Returns:
                - x, y: ints of top-left corner of agent's obs map.
            '''
            
            x, y, direction = agent_loc

            x, y = x + self.PADDING, y + self.PADDING

            x = x - (self.OBS_SIZE // 2)
            y = y - (self.OBS_SIZE // 2)


            x = jnp.where(direction == 0, x + (self.OBS_SIZE//2)-1, x)
            y = jnp.where(direction == 0, y, y)

            x = jnp.where(direction == 1, x, x)
            y = jnp.where(direction == 1, y + (self.OBS_SIZE//2)-1, y)


            x = jnp.where(direction == 2, x - (self.OBS_SIZE//2)+1, x)
            y = jnp.where(direction == 2, y, y)


            x = jnp.where(direction == 3, x, x)
            y = jnp.where(direction == 3, y - (self.OBS_SIZE//2)+1, y)
            return x, y

        def _get_obs(state: State) -> jnp.ndarray:
            '''
            Obtain the agent's observation of the grid.

            Args: 
                - state: State object containing env state.
            Returns:
                - jnp.ndarray of grid observation.
            '''
            # create state
            grid = jnp.pad(
                state.grid,
                ((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)),
                constant_values=Items.wall,
            )

            # obtain all agent obs-points
            agent_start_idxs = jax.vmap(_get_obs_point)(state.agent_locs)

            dynamic_slice = partial(
                jax.lax.dynamic_slice,
                operand=grid,
                slice_sizes=(self.OBS_SIZE, self.OBS_SIZE)
            )

            # obtain agent obs grids
            grids = jax.vmap(dynamic_slice)(start_indices=agent_start_idxs)

            # rotate agent obs grids
            grids = jax.vmap(rotate_grid)(state.agent_locs, grids)

            angles = jax.vmap(
                check_relative_orientation,
                in_axes=(0, None, 0)
            )(
                self._agents,
                state.agent_locs,
                grids
            )

            angles = jax.nn.one_hot(angles, 4)

            # one-hot (drop first channel as its empty blocks)
            grids = jax.nn.one_hot(
                grids - 1,
                num_agents + len(Items) - 1, # will be collapsed into a
                dtype=jnp.int8 # [Items, self, other, extra features] representation
            )

            # check agents that can interact
            inventory_sum = jnp.sum(state.agent_invs, axis=-1)
            agent_pickups = jnp.where(
                inventory_sum > INTERACT_THRESHOLD,
                True,
                False
            )

            # make index len(Item) always the current agent
            # and sum all others into an "other" agent
            grids = jax.vmap(
                combine_channels,
                in_axes=(0, 0, 0, None, None)
            )(
                grids,
                self._agents,
                angles,
                agent_pickups,
                state
            )

            return grids

        def get_current_s_interest(timestep):
            """Calculate current s_interest based on timestep and schedule."""
            if self.s_interest_schedule is None:
                return self.s_interest

            # Calculate which phase we're in using JAX operations
            phase = timestep // self.s_interest_change_every
            phase_idx = phase % self.s_interest_schedule.shape[0]
            return self.s_interest_schedule[phase_idx]

        def _interact_fire_zapping(
            key: jnp.ndarray, state: State, actions: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray, State, jnp.ndarray]:
            '''
            Main interaction logic entry point.

            Args:
                - key: jax key for randomisation.
                - state: State env state object.
                - actions: jnp.ndarray of actions taken by agents.
            Returns:
                - (jnp.ndarray, State, jnp.ndarray) - Tuple where index 0 is
                the array of rewards obtained, index 2 is the new env State,
                and index 3 is the new freeze penalty matrix.
            '''
            # if interact
            zaps = jnp.isin(actions,
                jnp.array(
                    [
                        Actions.zap_forward,
                        # Actions.zap_ahead
                    ]
                )
            )

            interact_idx = jnp.int16(Items.interact)

            # remove old interacts
            state = state.replace(grid=jnp.where(
                state.grid == interact_idx, jnp.int16(Items.empty), state.grid
            ))

            state = state.replace(grid=jnp.where(
                state.grid == Items.clean_beam, jnp.int16(Items.empty), state.grid
            ))

            # calculate pickups
            # agent_pickups = state.agent_invs.sum(axis=-1) > -100

            one_step_targets = jax.vmap(
                lambda p: p + STEP[p[2]]
            )(state.agent_locs)

            # check 2 ahead
            two_step_targets = jax.vmap(
                lambda p: p + 2*STEP[p[2]]
            )(state.agent_locs)


            target_right = jax.vmap(
                lambda p: p + STEP[p[2]] + STEP[(p[2] + 1) % 4]
            )(state.agent_locs)

            right_oob_check = jax.vmap(
                lambda t: jnp.logical_or(
                    jnp.logical_or((t[0] > self.GRID_SIZE_ROW - 1).any(), (t[1] > self.GRID_SIZE_COL - 1).any()),
                    (t < 0).any(),
                )
            )(target_right)

            target_right = jnp.where(
                right_oob_check[:, None],
                one_step_targets,
                target_right
            )


            target_left = jax.vmap(
                lambda p: p + STEP[p[2]] + STEP[(p[2] - 1) % 4]
            )(state.agent_locs)

            left_oob_check = jax.vmap(
                lambda t: jnp.logical_or(
                    jnp.logical_or((t[0] > self.GRID_SIZE_ROW - 1).any(), (t[1] > self.GRID_SIZE_COL - 1).any()),
                    (t < 0).any(),
                )
            )(target_left)

            target_left = jnp.where(
                left_oob_check[:, None],
                one_step_targets,
                target_left
            )

            # A beam cell outside the grid hits nothing. Without this, a negative index wraps to the far side of the
            # map (firing north from the top rows hit the bottom rows) and a too-large one clamps onto the edge. Park
            # such targets far out of range, where scatters drop them, and mask them out of every hit test.
            def _in_grid(t):
                return (t[:, 0] >= 0) & (t[:, 0] < self.GRID_SIZE_ROW) & (t[:, 1] >= 0) & (t[:, 1] < self.GRID_SIZE_COL)
            _off = jnp.array([self.GRID_SIZE_ROW + 100, self.GRID_SIZE_COL + 100, 0], dtype=one_step_targets.dtype)
            one_ok, two_ok, right_ok, left_ok = (_in_grid(t) for t in (one_step_targets, two_step_targets, target_right, target_left))
            one_step_targets = jnp.where(one_ok[:, None], one_step_targets, _off)
            two_step_targets = jnp.where(two_ok[:, None], two_step_targets, _off)
            target_right = jnp.where(right_ok[:, None], target_right, _off)
            target_left = jnp.where(left_ok[:, None], target_left, _off)
            beam_in_grid = jnp.concatenate((one_ok, two_ok, right_ok, left_ok), 0)

            all_zaped_locs = jnp.concatenate((one_step_targets, two_step_targets, target_right, target_left), 0)
            # zaps_3d = jnp.stack([zaps, zaps, zaps], axis=-1)

            zaps_4_locs = jnp.concatenate((zaps, zaps, zaps, zaps), 0) & beam_in_grid


            # all_zaped_locs = jax.vmap(filter_zaped_locs)(all_zaped_locs)

            def zaped_gird(a, z):
                return jnp.where(z, state.grid[a[0], a[1]], -1)

            all_zaped_gird = jax.vmap(zaped_gird)(all_zaped_locs, zaps_4_locs)
            # jax.debug.print("all_zaped_gird {all_zaped_gird}", all_zaped_gird=all_zaped_gird)

            def check_reborn_player(a):
                return jnp.isin(a, all_zaped_gird)
            
            reborn_players = jax.vmap(check_reborn_player)(self._agents)

            aux_grid = jnp.copy(state.grid)

            o_items = jnp.where(
                        state.grid[
                            one_step_targets[:, 0],
                            one_step_targets[:, 1]
                        ],
                        state.grid[
                            one_step_targets[:, 0],
                            one_step_targets[:, 1]
                        ],
                        interact_idx
                    )

            t_items = jnp.where(
                        state.grid[
                            two_step_targets[:, 0],
                            two_step_targets[:, 1]
                        ],
                        state.grid[
                            two_step_targets[:, 0],
                            two_step_targets[:, 1]
                        ],
                        interact_idx
                    )

            r_items = jnp.where(
                        state.grid[
                            target_right[:, 0],
                            target_right[:, 1]
                        ],
                        state.grid[
                            target_right[:, 0],
                            target_right[:, 1]
                        ],
                        interact_idx
                    )

            l_items = jnp.where(
                        state.grid[
                            target_left[:, 0],
                            target_left[:, 1]
                        ],
                        state.grid[
                            target_left[:, 0],
                            target_left[:, 1]
                        ],
                        interact_idx
                    )

            qualified_to_zap = zaps.reshape(-1)
            # jax.debug.print("qualified_to_zap {qualified_to_zap}", qualified_to_zap=qualified_to_zap)
            # update grid
            def update_grid(a_i, t, i, grid):
                return grid.at[t[:, 0], t[:, 1]].set(
                    jax.vmap(jnp.where)(
                        a_i,
                        i,
                        aux_grid[t[:, 0], t[:, 1]]
                    )
                )
            # def update_grid(a_i, t, i, grid):
            #     return grid.at[t[:, 0], t[:, 1]].set(2)


            # jax.debug.print("one_step_targets {one_step_targets}", one_step_targets=one_step_targets)
            aux_grid = update_grid(qualified_to_zap, one_step_targets, o_items, aux_grid)
            aux_grid = update_grid(qualified_to_zap, two_step_targets, t_items, aux_grid)
            aux_grid = update_grid(qualified_to_zap, target_right, r_items, aux_grid)
            aux_grid = update_grid(qualified_to_zap, target_left, l_items, aux_grid)

            # jax.debug.print("aux_grid {aux_grid}", aux_grid=aux_grid)
            state = state.replace(
                grid=jnp.where(
                    jnp.any(zaps),
                    aux_grid,
                    state.grid
                )
            )
            return reborn_players, state
        
        def _interact_fire_cleaning(
            key: jnp.ndarray, state: State, actions: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray, State, jnp.ndarray]:
            '''
            Main interaction logic entry point.

            Args:
                - key: jax key for randomisation.
                - state: State env state object.
                - actions: jnp.ndarray of actions taken by agents.
            Returns:
                - (jnp.ndarray, State, jnp.ndarray) - Tuple where index 0 is
                the array of rewards obtained, index 2 is the new env State,
                and index 3 is the new freeze penalty matrix.
            '''
            # if interact
            zaps = jnp.isin(actions,
                jnp.array(
                    [
                        Actions.zap_clean,
                    ]
                )
            )

            interact_idx = jnp.int16(Items.clean_beam)

            # remove old interacts

            state = state.replace(grid=jnp.where(
                state.grid == interact_idx, jnp.int16(Items.empty), state.grid
            ))


            one_step_targets = jax.vmap(
                lambda p: p + STEP[p[2]]
            )(state.agent_locs)

            two_step_targets = jax.vmap(
                lambda p: p + 2*STEP[p[2]]
            )(state.agent_locs)

            target_right = jax.vmap(
                lambda p: p + STEP[p[2]] + STEP[(p[2] + 1) % 4]
            )(state.agent_locs)

            right_oob_check = jax.vmap(
                lambda t: jnp.logical_or(
                    jnp.logical_or((t[0] > self.GRID_SIZE_ROW - 1).any(), (t[1] > self.GRID_SIZE_COL - 1).any()),
                    (t < 0).any(),
                )
            )(target_right)

            target_right = jnp.where(
                right_oob_check[:, None],
                one_step_targets,
                target_right
            )

            target_left = jax.vmap(
                lambda p: p + STEP[p[2]] + STEP[(p[2] - 1) % 4]
            )(state.agent_locs)

            left_oob_check = jax.vmap(
                lambda t: jnp.logical_or(
                    jnp.logical_or((t[0] > self.GRID_SIZE_ROW - 1).any(), (t[1] > self.GRID_SIZE_COL - 1).any()),
                    (t < 0).any(),
                )
            )(target_left)

            target_left = jnp.where(
                left_oob_check[:, None],
                one_step_targets,
                target_left
            )


            # A beam cell outside the grid hits nothing. Without this, a negative index wraps to the far side of the
            # map (firing north from the top rows hit the bottom rows) and a too-large one clamps onto the edge. Park
            # such targets far out of range, where scatters drop them, and mask them out of every hit test.
            def _in_grid(t):
                return (t[:, 0] >= 0) & (t[:, 0] < self.GRID_SIZE_ROW) & (t[:, 1] >= 0) & (t[:, 1] < self.GRID_SIZE_COL)
            _off = jnp.array([self.GRID_SIZE_ROW + 100, self.GRID_SIZE_COL + 100, 0], dtype=one_step_targets.dtype)
            one_ok, two_ok, right_ok, left_ok = (_in_grid(t) for t in (one_step_targets, two_step_targets, target_right, target_left))
            one_step_targets = jnp.where(one_ok[:, None], one_step_targets, _off)
            two_step_targets = jnp.where(two_ok[:, None], two_step_targets, _off)
            target_right = jnp.where(right_ok[:, None], target_right, _off)
            target_left = jnp.where(left_ok[:, None], target_left, _off)
            beam_in_grid = jnp.concatenate((one_ok, two_ok, right_ok, left_ok), 0)

            all_zaped_locs = jnp.concatenate((one_step_targets, two_step_targets, target_right, target_left), 0)
            # zaps_3d = jnp.stack([zaps, zaps, zaps], axis=-1)

            zaps_4_locs_judge = jnp.concatenate((zaps, zaps, zaps, zaps), 0) & beam_in_grid


            # all_zaped_locs = jax.vmap(filter_zaped_locs)(all_zaped_locs)

            # Whether a cell clears at all is dictated by the hidden rule: the
            # beam only works on a waste type when the firing agent happens to
            # be holding that type's tool, and does nothing whatsoever
            # otherwise. Making the wrong tool useless rather than merely
            # slower is what forces the rule to be inferred — an earlier
            # version only slowed it down, and indiscriminate cleaning then
            # scored within a few percent of an oracle.
            # The beam footprint is four blocks of one entry per agent, so
            # tiling the held-tool vector the same way lines each target up
            # with the agent that shot it.
            tool_per_target = jnp.tile(state.held_tool, 4)

            def clean_gird(a, judge):
                cur = state.grid[a[:, 0], a[:, 1]]
                idx = jnp.clip(cur - DIRT_TYPES[0], 0, NUM_DIRT_TYPES - 1)
                # Interpret the encoded ruleset per target cell, XLand style:
                # check_clean scans the rows and switches on rule_id, so this
                # line never changes when new rule types are added to the
                # vocabulary. The is_dirt gate below keeps the clipped idx of
                # non-waste cells from mattering.
                # apply_beam returns what the cell BECOMES: water for a
                # ToolClearsRule, another colour for a ToolTransformsRule
                # (a chain link), NO_EFFECT when no row fires.
                result = jax.vmap(apply_beam, in_axes=(None, 0, 0, None, None))(
                    state.rule_encoding, idx, tool_per_target,
                    int(Items.potential_dirt), DIRT_BASE,
                ).astype(jnp.int16)
                if self.craft_rules and not self.attr_craft:
                    # the crafted tool clears waste of any colour; without it
                    # the beam does nothing (the attribute-craft variant goes
                    # through apply_beam: its crafted ids are the tools the
                    # Hue / Shade rows name, base tools match nothing)
                    crafted_t = jnp.tile(state.crafted, 4)
                    fired = (judge == True) & is_dirt(cur) & crafted_t
                    result = jnp.where(fired, jnp.int16(Items.potential_dirt), cur)
                    will_clear = fired
                else:
                    fired = (judge == True) & is_dirt(cur) & (result != NO_EFFECT)
                    will_clear = fired & (result == jnp.int16(Items.potential_dirt))
                # Write ONLY the targets that fired. `a` holds every agent's footprint, fired or not, and a
                # scatter with repeated indices keeps one write arbitrarily (the last on CPU, unspecified on
                # GPU). Writing `cur` back for the entries that did not fire therefore let a bystander whose
                # footprint happened to cover a cell erase another agent's shot on it: an agent merely walking
                # up beside the river could cancel a clean. Non-firing entries are now sent off the grid and
                # dropped, so a cell changes exactly when some beam that fired on it says so.
                off_r = jnp.int16(self.GRID_SIZE_ROW + 100)
                off_c = jnp.int16(self.GRID_SIZE_COL + 100)
                rows = jnp.where(fired, a[:, 0], off_r)
                cols = jnp.where(fired, a[:, 1], off_c)
                grid = state.grid.at[rows, cols].set(result.astype(state.grid.dtype), mode="drop")
                # the rule's verdict on every target, as it was applied: what the cell held, whether this
                # beam acted on it, and what it became. NO_EFFECT where the tool does nothing to that cell.
                verdict = jnp.where(fired, result, jnp.int16(NO_EFFECT))
                return grid, will_clear, cur, verdict


            judge_flat = zaps_4_locs_judge.reshape(-1)
            grid_clean, will_clear, beam_before, beam_verdict = clean_gird(all_zaped_locs, judge_flat)
            # Cells this agent's own beam cleared this step (a diagnostic, not
            # a reward). Targets come in four blocks of one entry per agent,
            # so folding the block axis gives per-agent counts. At map edges
            # an out-of-bounds side target falls back to the front cell and a
            # clear can be counted twice; harmless for a diagnostic.
            cleaned_by_agent = will_clear.reshape(4, -1).sum(axis=0)
            # What each agent's beam did, target by target, straight from the rule interpreter above: the
            # environment's own account of a shot, for callers that want it (report_beam=True). Targets come
            # in four blocks of one entry per agent (front, two ahead, right, left); transposed to (agent, 4).
            n_ag = zaps.shape[0]
            beam_report = {
                "beam_rc": all_zaped_locs[:, :2].reshape(4, n_ag, 2).transpose(1, 0, 2).astype(jnp.int16),
                "beam_hit": judge_flat.reshape(4, n_ag).T,
                "beam_before": beam_before.reshape(4, n_ag).T.astype(jnp.int16),
                "beam_after": beam_verdict.reshape(4, n_ag).T.astype(jnp.int16),
            }
            state = state.replace(grid=grid_clean)

            # refresh label

            def renew_dirt_label(locs, labels):
                return jnp.where(is_dirt(grid_clean[locs[0], locs[1]]) | (grid_clean[locs[0], locs[1]] == Items.potential_dirt), grid_clean[locs[0], locs[1]], labels)


            renew_label = jax.vmap(renew_dirt_label)(state.potential_dirt_and_dirt_locs, state.potential_dirt_and_dirt_label)


            state = state.replace(
                potential_dirt_and_dirt_label=renew_label
            )

            
            aux_grid = jnp.copy(state.grid)

            o_items = jnp.where(
                        state.grid[
                            one_step_targets[:, 0],
                            one_step_targets[:, 1]
                        ],
                        state.grid[
                            one_step_targets[:, 0],
                            one_step_targets[:, 1]
                        ],
                        interact_idx
                    )

            t_items = jnp.where(
                        state.grid[
                            two_step_targets[:, 0],
                            two_step_targets[:, 1]
                        ],
                        state.grid[
                            two_step_targets[:, 0],
                            two_step_targets[:, 1]
                        ],
                        interact_idx
                    )

            r_items = jnp.where(
                        state.grid[
                            target_right[:, 0],
                            target_right[:, 1]
                        ],
                        state.grid[
                            target_right[:, 0],
                            target_right[:, 1]
                        ],
                        interact_idx
                    )

            l_items = jnp.where(
                        state.grid[
                            target_left[:, 0],
                            target_left[:, 1]
                        ],
                        state.grid[
                            target_left[:, 0],
                            target_left[:, 1]
                        ],
                        interact_idx
                    )

            qualified_to_zap = zaps.reshape(-1)


            # update grid
            def update_grid(a_i, t, i, grid):
                return grid.at[t[:, 0], t[:, 1]].set(
                    jax.vmap(jnp.where)(
                        a_i,
                        i,
                        aux_grid[t[:, 0], t[:, 1]]
                    )
                )



            aux_grid = update_grid(qualified_to_zap, one_step_targets, o_items, aux_grid)
            aux_grid = update_grid(qualified_to_zap, two_step_targets, t_items, aux_grid)
            aux_grid = update_grid(qualified_to_zap, target_right, r_items, aux_grid)
            aux_grid = update_grid(qualified_to_zap, target_left, l_items, aux_grid)


            state = state.replace(
                grid=jnp.where(
                    jnp.any(zaps),
                    aux_grid,
                    state.grid
                )
            )
            return state, cleaned_by_agent, beam_report


        def _step(
            key: chex.PRNGKey,
            state: State,
            actions: jnp.ndarray,
            timestep: int = 0
        ):
            """Step the environment."""

            # regrowth of apply
            grid_apple = state.grid
            # Every waste type pollutes, exactly as the single waste type does
            # in vanilla Clean Up. What the hidden rule decides is which TOOL
            # clears a type, not whether the type matters — so the pollution
            # accounting is left untouched and thresholdDepletion stays
            # calibrated as inherited.
            labels = state.potential_dirt_and_dirt_label
            dirtCount = jnp.sum(is_dirt(labels))
            totalDirtCount = dirtCount
            dirtFraction = dirtCount / (
                len(state.potential_dirt_and_dirt_locs) + len(self.RIVER))
            depletion = self.thresholdDepletion
            restoration = self.thresholdRestoration
            interpolation = (dirtFraction - depletion) / (restoration - depletion)

            interpolation = jnp.clip(interpolation, -jnp.inf, 1.0)
            probability = self.maxAppleGrowthRate * interpolation
            def regrow_apple(apple_locs, p):
                new_apple = jnp.where((((grid_apple[apple_locs[0], apple_locs[1]] == Items.empty) & (p < probability)) 
                                       | ((grid_apple[apple_locs[0], apple_locs[1]] == Items.apple))),  
                                      Items.apple, Items.empty)
                return new_apple
            prob = jax.random.uniform(key, shape=(len(self.POTENTIAL_APPLE),))
            apple_cells = _layout(state.layout_id)["apple"]
            new_apple = jax.vmap(regrow_apple)(apple_cells, prob)

            new_apple_grid = grid_apple.at[apple_cells[:, 0], apple_cells[:, 1]].set(new_apple)
            state = state.replace(grid=new_apple_grid)

            # DirtSpawning update the grid and potential_dirt_and_dirt_label
            grid_dirt = state.grid

            noise = jax.random.uniform(key, shape=(len(state.potential_dirt_and_dirt_label),)) * 1e-4
            label_with_noise = state.potential_dirt_and_dirt_label + noise

            label_with_noise_rank = jnp.sort(label_with_noise)
            unstable_indices = jnp.argsort(label_with_noise)

            unstable_sorted_locs = state.potential_dirt_and_dirt_locs[unstable_indices]
            
            p = jax.random.uniform(key, shape=(1,))
            key, dirt_type_key = jax.random.split(key)
            spawn_type = DIRT_TYPES[jax.random.randint(dirt_type_key, (), 0, NUM_DIRT_TYPES)]
            if self.skewed_waste:
                # respawns follow the trial's dominant type at the same skew,
                # so the trial-level composition holds up over time instead
                # of washing back to uniform
                key, skew_key = jax.random.split(key)
                if self.attr_rules:
                    # dom_type is a hue: pick one of its shades
                    key, shade_key = jax.random.split(key)
                    dom_code = DIRT_TYPES[state.dom_type * NUM_SHADES
                                          + jax.random.randint(shade_key, (), 0, NUM_SHADES)]
                else:
                    dom_code = DIRT_TYPES[state.dom_type]
                spawn_type = jnp.where(
                    jax.random.uniform(skew_key, ()) < self.waste_skew,
                    dom_code, spawn_type)
            one_piece_dirt = jnp.where(((grid_dirt[unstable_sorted_locs[0, 0], unstable_sorted_locs[0, 1]] == Items.potential_dirt) 
                                       & (p < self.dirtSpawnProbability) & (state.inner_t>self.delayStartOfDirtSpawning)),  
                        spawn_type, label_with_noise_rank[0])

            label_with_noise_rank_new = label_with_noise_rank.at[0].set(one_piece_dirt[0]) 

            label_rank_new = jnp.round(label_with_noise_rank_new).astype(jnp.int16)
            

            state = state.replace(potential_dirt_and_dirt_label=label_rank_new)
            state = state.replace(potential_dirt_and_dirt_locs=unstable_sorted_locs)
            actions = jnp.array(actions)

            new_grid = state.grid.at[
                state.agent_locs[:, 0],
                state.agent_locs[:, 1]
            ].set(
                jnp.int16(Items.empty)
            )

            new_grid = new_grid.at[state.potential_dirt_and_dirt_locs[:, 0], state.potential_dirt_and_dirt_locs[:, 1]].set(state.potential_dirt_and_dirt_label)
            
            new_grid = new_grid.at[_layout(state.layout_id)["river"][:, 0], _layout(state.layout_id)["river"][:, 1]].set(Items.river)



            x, y = state.reborn_locs[:, 0], state.reborn_locs[:, 1]
            new_grid = new_grid.at[x, y].set(self._agents)
            state = state.replace(grid=new_grid)
            state = state.replace(agent_locs=state.reborn_locs)

            key, subkey = jax.random.split(key)
            all_new_locs = jax.vmap(lambda p, a: jnp.int16(p + ROTATIONS[a]) % jnp.array([self.GRID_SIZE_ROW + 1, self.GRID_SIZE_COL + 1, 4], dtype=jnp.int16))(p=state.agent_locs, a=actions).reshape(self.num_agents, 3)
            # (explicit reshape, not squeeze: with num_agents=1 a squeeze
            # would also drop the agent axis and desync every later vmap)

            agent_move = (actions == Actions.up) | (actions == Actions.down) | (actions == Actions.right) | (actions == Actions.left)
            all_new_locs = jax.vmap(lambda m, n, p: jnp.where(m, n + STEP_MOVE[p], n))(m=agent_move, n=all_new_locs, p=actions)
            
            all_new_locs = jax.vmap(
                jnp.clip,
                in_axes=(0, None, None)
            )(
                all_new_locs,
                jnp.array([0, 0, 0], dtype=jnp.int16),
                jnp.array(
                    [self.GRID_SIZE_ROW - 1, self.GRID_SIZE_COL - 1, 3],
                    dtype=jnp.int16
                ),
            ).reshape(self.num_agents, 3)

            # Resolve agent-agent movement conflicts via the shared
            # resolver: same-target conflicts (non-mover priority, else
            # random winner), swaps blocked, trains allowed, cascades
            # reverted to a fixed point. Final (row, col) unique.
            key, move_key = jax.random.split(key)
            new_locs = resolve_movement(move_key, state.agent_locs, all_new_locs)

            # get apples
            def coin_matcher(p: jnp.ndarray) -> jnp.ndarray:
                c_matches = jnp.array([
                    state.grid[p[0], p[1]] == Items.apple
                    ])
                return c_matches
            
            apple_matches = jax.vmap(coin_matcher)(p=new_locs)

            # # individual rewards
            # rewards = jnp.zeros((self.num_agents, 1))
            # rewards = jnp.where(apple_matches, 1, rewards)

            # # single reward or sum reward

            # rewards_sum_all_agents = jnp.zeros((self.num_agents, 1))
            # rewards_sum = jnp.sum(rewards)
            # rewards_sum_all_agents += rewards_sum
            # rewards = rewards_sum_all_agents

            new_invs = state.agent_invs + apple_matches

            state = state.replace(
                agent_invs=new_invs
            )

            # update grid
            old_grid = state.grid

            new_grid = old_grid.at[
                state.agent_locs[:, 0],
                state.agent_locs[:, 1]
            ].set(
                jnp.int16(Items.empty)
            )

            new_grid = new_grid.at[state.potential_dirt_and_dirt_locs[:, 0], state.potential_dirt_and_dirt_locs[:, 1]].set(state.potential_dirt_and_dirt_label)

            new_grid = new_grid.at[_layout(state.layout_id)["river"][:, 0], _layout(state.layout_id)["river"][:, 1]].set(Items.river)
            # Tool cells are not consumed: re-stamp them so one reappears once
            # the agent standing on it walks away.
            station_cells = _layout(state.layout_id)["tools"]
            new_grid = new_grid.at[station_cells[:, 0], station_cells[:, 1]].set(
                self._tool_labels)
            x, y = new_locs[:, 0], new_locs[:, 1]
            new_grid = new_grid.at[x, y].set(self._agents)
            state = state.replace(grid=new_grid)

            # Stepping onto a tool swaps what the agent carries. Picking up
            # costs no action of its own — with three tools to try and a rule
            # to infer, an extra pickup action to learn first would only get
            # in the way.
            stepped = _layout(state.layout_id)["tool_grid"][new_locs[:, 0], new_locs[:, 1]]
            # explicit_pickup: a pickup is the pick_up action while standing on
            # a station; otherwise it is ENTERING the station.
            picking = (actions.reshape(-1) == Actions.pick_up) if self.explicit_pickup else None
            if self.craft_rules:
                # A pickup is ENTERING a station (standing on one does not
                # repeat it). It is appended to the sliding window; when the
                # window equals the hidden order the agent has crafted the
                # tool, and stations stop mattering for it.
                moved = jnp.any(new_locs[:, :2] != state.agent_locs[:, :2], axis=-1)
                if self.attr_craft:
                    # one hand: every station entered counts (a crafted tool
                    # is replaced by the base tool of the next station), and
                    # a window equal to some recipe hands over its product
                    entered = (stepped >= 0) & (picking if self.explicit_pickup else moved)
                    window = jnp.concatenate(
                        [state.pickups[:, 1:], stepped[:, None].astype(jnp.int16)], axis=1)
                    pickups = jnp.where(entered[:, None], window, state.pickups)
                    valid, products, seqs = decode_craft_recipes(state.rule_encoding, self.craft_len)
                    match = valid[None, :] & jnp.all(pickups[:, None, :] == seqs[None, :, :], axis=-1)
                    product = products[jnp.argmax(match, axis=1)]
                    new_held = jnp.where(entered,
                                         jnp.where(match.any(axis=1), product, stepped),
                                         state.held_tool).astype(jnp.int16)
                    state = state.replace(
                        agent_locs=new_locs,
                        held_tool=new_held,
                        pickups=pickups.astype(jnp.int16),
                        crafted=new_held >= self.craft_tools,
                    )
                else:
                    entered = (stepped >= 0) & (picking if self.explicit_pickup else moved) & ~state.crafted
                    window = jnp.concatenate(
                        [state.pickups[:, 1:], stepped[:, None].astype(jnp.int16)], axis=1)
                    pickups = jnp.where(entered[:, None], window, state.pickups)
                    secret = decode_craft_sequence(state.rule_encoding, self.craft_len)
                    crafted = state.crafted | (entered & jnp.all(pickups == secret[None, :], axis=-1))
                    state = state.replace(
                        agent_locs=new_locs,
                        held_tool=jnp.where(entered, stepped, state.held_tool).astype(jnp.int16),
                        pickups=pickups.astype(jnp.int16),
                        crafted=crafted,
                    )
            else:
                take = (stepped >= 0) & (picking if self.explicit_pickup else True)
                state = state.replace(
                    agent_locs=new_locs,
                    held_tool=jnp.where(take, stepped,
                                        state.held_tool).astype(jnp.int16),
                )

            reborn_players, state = _interact_fire_zapping(key, state, actions)

            state, cleaned_by_agent, beam_report = _interact_fire_cleaning(
                key, state, actions)

            # Occupancy-aware respawn: reborn agents are placed on spawn
            # cells not occupied by any survivor, so no overlap is possible.
            key, respawn_key = jax.random.split(key)
            new_re_locs = resolve_respawn(
                respawn_key, new_locs, reborn_players.astype(bool), _layout(state.layout_id)["spawns"]
            )
            state = state.replace(reborn_locs=new_re_locs)

            # Rewards are individual only: an apple pays the agent that ate it. There is deliberately no
            # common-reward mode -- summing everyone's apples would remove the social dilemma.
            if self.inequity_aversion:
                rewards = jnp.zeros((self.num_agents, 1))
                original_rewards = jnp.where(apple_matches, 1, rewards) * self.num_agents
                if self.smooth_rewards:
                    should_smooth = (state.inner_t % 1) == 0
                    new_smooth_rewards = 0.99 * 0.01* state.smooth_rewards + original_rewards
                    rewards, disadvantageous, advantageous = self.get_inequity_aversion_rewards_immediate(new_smooth_rewards, state.inner_t, self.inequity_aversion_target_agents, self.inequity_aversion_alpha, self.inequity_aversion_beta)
                    state = state.replace(smooth_rewards=new_smooth_rewards)
                    info = {
                    "original_rewards": original_rewards.squeeze(),
                    "smooth_rewards": state.smooth_rewards.squeeze(),
                }
                else:
                    rewards, disadvantageous, advantageous = self.get_inequity_aversion_rewards_immediate(original_rewards, state.inner_t, self.inequity_aversion_target_agents, self.inequity_aversion_alpha, self.inequity_aversion_beta)
                    info = {
                    "original_rewards": original_rewards.squeeze(),
                }
            elif self.svo:
                rewards = jnp.zeros((self.num_agents, 1))
                original_rewards = jnp.where(apple_matches, 1, rewards) * self.num_agents
                rewards, theta = self.get_svo_rewards(original_rewards, self.svo_w, self.svo_ideal_angle_degrees, self.svo_target_agents)
                info = {
                    "original_rewards": original_rewards.squeeze(),
                    "svo_theta": theta.squeeze(),
                }
            elif self.interest:
                rewards = jnp.zeros((self.num_agents, 1))
                original_rewards = jnp.where(apple_matches, 1, rewards) * self.num_agents
                original_flat = original_rewards.squeeze()

                # Calculate current s_interest based on timestep
                current_s_interest = get_current_s_interest(timestep)

                # Each agent gets s * their_reward + (1-s)/(n-1) * sum_of_others
                total_reward = jnp.sum(original_flat)
                others_reward = total_reward - original_flat  # sum of all other agents' rewards

                rewards = (current_s_interest * original_flat +
                        (1 - current_s_interest) / (self.num_agents - 1) * others_reward).reshape(-1, 1)

                info = {
                    "original_rewards": original_rewards.squeeze(),
                    "s_interest": current_s_interest,
                }
            elif self.cf:
                rewards = jnp.zeros((self.num_agents, 1))
                original_rewards = jnp.where(apple_matches, 1, rewards) * self.num_agents
                rewards, theta = self.get_cf_rewards(original_rewards, self.cf_w, self.cf_ideal_angle_degrees, self.cf_target_agents)
                info = {
                    "original_rewards": original_rewards.squeeze(),
                    "cf_theta": theta.squeeze(),
                }
            else:
                rewards = jnp.zeros((self.num_agents, 1))
                rewards = jnp.where(apple_matches, 1, rewards) * self.num_agents
                info = {
                    "original_rewards": rewards.squeeze(),
                }
            
            # Cells cleared by each agent's own beam this step — a diagnostic
            # (and the auxiliary head's label), never a reward: the policy is
            # trained on apples only.
            info["cleaned_by_agent"] = cleaned_by_agent.squeeze()
            if self.report_beam:
                # opt-in: these are (agent, 4[, 2]) and the RL loops reshape every info leaf to one value
                # per actor, so they would break on them. The LLM harness turns this on and builds its
                # rule ledger from it instead of diffing grids.
                info.update(beam_report)
            info["crafted"] = state.crafted.astype(jnp.int16).reshape(-1).squeeze()
            info["clean_action_info"] = jnp.where(actions == Actions.zap_clean, 1, 0).squeeze()
            info["cleaned_water"] = jnp.array([len(state.potential_dirt_and_dirt_label) - dirtCount] * self.num_agents).squeeze()
            info["waste_cleared"] = jnp.array([len(state.potential_dirt_and_dirt_label) - dirtCount] * self.num_agents).squeeze()
            # Hidden-rule diagnostics. active_waste is the pollution that
            # actually matters; inactive_waste is what cleaning it would waste
            # effort on. Together they show whether the population has found
            # the hidden rule, and who paid the exploration cost.
            info["active_waste"] = jnp.array([dirtCount] * self.num_agents).squeeze()
            info["inactive_waste"] = jnp.array(
                [totalDirtCount - dirtCount] * self.num_agents).squeeze()
            # which trial of the meta-episode this is, so per-trial learning
            # curves can be plotted (does trial 2 beat trial 1?)
            info["trial"] = jnp.array([state.outer_t] * self.num_agents).squeeze()
            # Diagnostics read the ruleset through the dense-table projection;
            # the keys keep their historical meaning (the tool clearing waste
            # types 0..2) so old and new runs stay comparable in wandb.
            rule_table = decode_tool_table(state.rule_encoding, NUM_DIRT_TYPES)
            info["rule_type"] = jnp.array([rule_table[0]] * self.num_agents).squeeze()
            info["rule_arg_a"] = jnp.array([rule_table[1]] * self.num_agents).squeeze()
            info["rule_arg_b"] = jnp.array([rule_table[2]] * self.num_agents).squeeze()
            
            state_nxt = State(
                agent_locs=state.agent_locs,
                agent_invs=state.agent_invs,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
                grid=state.grid,
                apples=state.apples,
                freeze=state.freeze,
                reborn_locs=state.reborn_locs,
                potential_dirt_and_dirt_locs=state.potential_dirt_and_dirt_locs,
                potential_dirt_and_dirt_label=state.potential_dirt_and_dirt_label,
                smooth_rewards=state.smooth_rewards,
                rule_encoding=state.rule_encoding,
                held_tool=state.held_tool,
                dom_type=state.dom_type,
                pickups=state.pickups,
                crafted=state.crafted,
                layout_id=state.layout_id,
            )

            # now calculate if done for inner or outer episode
            inner_t = state_nxt.inner_t
            outer_t = state_nxt.outer_t
            reset_inner = inner_t == num_inner_steps

            # If the trial is over, restart from a fresh layout — but keep
            # this meta-episode's hidden rule. Trials are how an agent gets to
            # USE what it worked out: the rule only changes when the whole
            # meta-episode ends (and the base class autoresets), so evidence
            # gathered in trial 1 is still valid in trial 2.
            state_re = _reset_state(
                key, state.dom_type if self.dom_hue_per_episode else None,
                layout_override=state.layout_id)   # the map, like the rules, stays for the whole meta-episode

            # The tool an agent carries survives the trial boundary too: it
            # is evidence, not scenery, and dropping it would force the search
            # to restart from nothing every trial.
            if self.craft_rules:
                # the crafted tool (and the pickup window) is lost with the
                # map; only the hidden order survives the trial boundary
                state_re = state_re.replace(
                    outer_t=outer_t + 1,
                    rule_encoding=state_nxt.rule_encoding,
                )
            else:
                state_re = state_re.replace(
                    outer_t=outer_t + 1,
                    rule_encoding=state_nxt.rule_encoding,
                    held_tool=state_nxt.held_tool,
                )
            state = jax.tree.map(
                lambda x, y: jnp.where(reset_inner, x, y),
                state_re,
                state_nxt,
            )
            outer_t = state.outer_t
            reset_outer = outer_t == num_outer_steps
            done = {f'{a}': reset_outer for a in self.agents}
            # done = [reset_outer for _ in self.agents]
            done["__all__"] = reset_outer

            obs = _get_obs(state)
            rewards = jnp.where(
                reset_inner,
                jnp.zeros_like(rewards, dtype=jnp.int16),
                rewards
            )

            # mean_inv = state.agent_invs.mean(axis=0)
            return (
                obs,
                state,
                rewards.reshape(-1),   # (num_agents,) even for one agent —
                                       # squeeze() collapsed it to a scalar
                done,
                info,
            )

        def _reset_state(
            key: jnp.ndarray,
            dom_override=None,   # traced dom_type to reuse (dom_hue_per_episode)
            layout_override=None,   # traced layout_id to reuse (trial boundary)
        ) -> State:
            if self.layout_pool is None:
                layout_id = jnp.int16(0)
            elif layout_override is not None:
                layout_id = layout_override
            else:   # own key stream (fold_in), so every draw below is the same as without a pool
                layout_id = jax.random.randint(jax.random.fold_in(key, 0x1A7), (), 0, self.layout_pool.size).astype(jnp.int16)
            L = _layout(layout_id)
            key, subkey = jax.random.split(key)

            # Find the free spaces in the grid
            grid = jnp.zeros((self.GRID_SIZE_ROW, self.GRID_SIZE_COL), jnp.int16)


            inside_players_pos = jax.random.permutation(subkey, L["spawns_in"])
            player_positions = jnp.concatenate((inside_players_pos, L["spawns"]))
            agent_pos = jax.random.permutation(subkey, player_positions)[:num_agents]
            wall_pos = L["walls"]
            apple_pos = L["apple"]

            river = L["river"]
            potential_dirt = L["potential_dirt"]
            dirt = L["dirt"]

            potential_dirt_label = jnp.zeros((len(potential_dirt)), dtype=jnp.int16) +Items.potential_dirt
            # Initial waste mixes all six types, so no type can be ruled out
            # by its abundance alone.
            key, dirt_key, rule_key, tool_key = jax.random.split(key, 4)
            dirt_label = DIRT_TYPES[jax.random.randint(dirt_key, (len(dirt),), 0, NUM_DIRT_TYPES)]
            # Only `initial_dirt` of the waste cells open dirty; the rest open
            # as clean water and fill in at the usual spawn rate.
            if self.initial_dirt_range is not None:
                lo, hi = self.initial_dirt_range
                key, keep_key, frac_key = jax.random.split(key, 3)
                frac = jax.random.uniform(frac_key, (), minval=lo, maxval=hi)
                keep = jax.random.uniform(keep_key, (len(dirt),)) < frac
                dirt_label = jnp.where(keep, dirt_label,
                                       jnp.int16(Items.potential_dirt))
            elif self.initial_dirt < 1.0:
                key, keep_key = jax.random.split(key)
                keep = jax.random.uniform(keep_key, (len(dirt),)) < self.initial_dirt
                dirt_label = jnp.where(keep, dirt_label,
                                       jnp.int16(Items.potential_dirt))
            # The meta-episode's hidden ruleset, as a padded XLand-style
            # encoding. Drawn from the benchmark pool when one is supplied, so
            # held-out rulesets can be kept out of training entirely; the pool
            # sample is an array index, so under vmap over reset keys every
            # parallel environment draws its own task.
            if self.rule_pool is not None:
                rule_encoding = self.rule_pool.sample(rule_key)
            elif self.attr_craft and self.per_colour_tools:
                rule_encoding = sample_type_craft_ruleset(rule_key, self.craft_tools, self.craft_len, NUM_DIRT_COLORS)
            elif self.attr_craft:
                rule_encoding = sample_attr_craft_ruleset(
                    rule_key, self.craft_tools, self.craft_len, NUM_HUES, NUM_SHADES)
            elif self.attr_rules:
                rule_encoding = sample_attr_ruleset(
                    rule_key, self.num_tools, NUM_HUES, NUM_SHADES)
            elif self.craft_rules:
                rule_encoding = sample_craft_ruleset(rule_key, self.num_tools, self.craft_len)
            elif self.bijective_rules:
                rule_encoding = sample_bijective_ruleset(rule_key)
            elif self.chain_depth >= 2:
                rule_encoding = sample_chain_ruleset(
                    rule_key, self.chain_depth, self.p_chain)
            else:
                rule_encoding = sample_ruleset(rule_key)
            # Per-trial dominant waste type. The extra key splits live inside
            # the static branch so the default mode's RNG stream — and with
            # it every historical result — is untouched.
            if self.skewed_waste:
                key, dom_key, mix_key = jax.random.split(key, 3)
                if self.attr_rules:
                    # the dominant thing is a HUE; its shades are uniform, so
                    # no single tool can clear more than a third of the river
                    key, shade_key = jax.random.split(key)
                    dom_type = jax.random.randint(
                        dom_key, (), 0, NUM_HUES).astype(jnp.int16)
                    if self.fixed_dom_type is not None:    # the key is still split, so the stream below is unchanged
                        dom_type = jnp.int16(self.fixed_dom_type)
                    if dom_override is not None:
                        dom_type = dom_override
                    shade = jax.random.randint(shade_key, (len(dirt),), 0, NUM_SHADES)
                    dom_code = DIRT_TYPES[dom_type * NUM_SHADES + shade]
                else:
                    dom_type = jax.random.randint(
                        dom_key, (), 0, NUM_DIRT_TYPES).astype(jnp.int16)
                    if self.fixed_dom_type is not None:
                        dom_type = jnp.int16(self.fixed_dom_type)
                    if dom_override is not None:
                        dom_type = dom_override
                    dom_code = DIRT_TYPES[dom_type]
                take_dom = (jax.random.uniform(mix_key, (len(dirt),))
                            < self.waste_skew)
                # only re-type cells that are actually waste (respect the
                # initial_dirt curriculum's clean-water holes)
                dirt_label = jnp.where(
                    take_dom & is_dirt(dirt_label), dom_code, dirt_label)
            else:
                dom_type = jnp.int16(0)
            # Agents start holding a random tool rather than empty-handed:
            # otherwise the first thing to learn is "walk to a tool", before
            # any reward at all can be collected.
            held_tool = jax.random.randint(
                tool_key, (num_agents,), 0, self.num_tools).astype(jnp.int16)

            potential_dirt_and_dirt = jnp.concatenate((potential_dirt, dirt))
            potential_dirt_and_dirt_label = jnp.concatenate((potential_dirt_label, dirt_label))


            # set wall
            grid = grid.at[
                wall_pos[:, 0],
                wall_pos[:, 1]
            ].set(jnp.int16(Items.wall))

            # set dirt (each cell keeps the type drawn above)
            grid = grid.at[dirt[:, 0],
                           dirt[:, 1]
                           ].set(dirt_label)
            
            # set river
            grid = grid.at[river[:, 0],
                            river[:, 1]
                            ].set(jnp.int16(Items.river))
            
            # set potential dirt
            grid = grid.at[potential_dirt[:, 0],
                            potential_dirt[:, 1]
                            ].set(jnp.int16(Items.potential_dirt))

            # set the tool stations
            grid = grid.at[L["tools"][:, 0],
                           L["tools"][:, 1]].set(self._tool_labels)
            


            player_dir = jax.random.randint(
                subkey, shape=(
                    num_agents,
                    ), minval=0, maxval=3, dtype=jnp.int8
            )

            agent_locs = jnp.array(
                [agent_pos[:, 0], agent_pos[:, 1], player_dir],
                dtype=jnp.int16
            ).T

            grid = grid.at[
                agent_locs[:, 0],
                agent_locs[:, 1]
            ].set(jnp.int16(self._agents))

            freeze = jnp.array(
                [[-1]*num_agents]*num_agents,
            dtype=jnp.int16
            )

            return State(
                agent_locs=agent_locs,
                agent_invs=jnp.array([(0,0)]*num_agents, dtype=jnp.int8),
                inner_t=0,
                outer_t=0,
                grid=grid,
                apples=apple_pos,

                freeze=freeze,
                reborn_locs=agent_locs,
                potential_dirt_and_dirt_locs=potential_dirt_and_dirt,
                potential_dirt_and_dirt_label=potential_dirt_and_dirt_label,
                smooth_rewards=jnp.zeros((self.num_agents, 1)),
                rule_encoding=rule_encoding,
                held_tool=held_tool,
                dom_type=dom_type,
                pickups=jnp.full((num_agents, self.craft_len), -1, dtype=jnp.int16),
                crafted=jnp.zeros((num_agents,), dtype=bool),
                layout_id=layout_id,
            )

        def reset(
            key: jnp.ndarray
        ) -> Tuple[jnp.ndarray, State]:
            state = _reset_state(key)
            obs = _get_obs(state)
            return obs, state
        
        ################################################################################
        # if you want to test whether it can run on gpu, activate following code
        # overwrite Gymnax as it makes single-agent assumptions
        if jit:
            self.step_env = jax.jit(_step)
            self.reset = jax.jit(reset)
            self.get_obs_point = jax.jit(_get_obs_point)
        else:
            # if you want to see values whilst debugging, don't jit
            self.step_env = _step
            self.reset = reset
            self.get_obs_point = _get_obs_point
        ################################################################################

    def layout_arrays(self, state):
        """Host-side (numpy) cell arrays of the map `state` plays on: river / potential_dirt / dirt / apple / spawns /
        tools / tool_grid, plus the sets river_set (stream) and orchard_set. Cached per layout."""
        lid = int(onp.asarray(state.layout_id)) if self.layout_pool is not None else 0
        if lid not in self._layout_host_cache:
            d = {k: onp.asarray(v) for k, v in self._layout(lid).items()}
            d["river_set"] = set(map(tuple, d["river"].tolist()))
            d["orchard_set"] = set(map(tuple, d["apple"].tolist()))
            d["waste_cells"] = onp.concatenate([d["potential_dirt"], d["dirt"]]).astype(onp.int64)
            self._layout_host_cache[lid] = d
        return self._layout_host_cache[lid]

    @property
    def name(self) -> str:
        """Environment name."""
        return "MGinTheGrid"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment (pick_up only with explicit_pickup)."""
        return len(Actions) if self.explicit_pickup else len(Actions) - 1

    def action_space(
        self, agent_id: Union[int, None] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self) -> spaces.Dict:
        """Observation space of the environment."""
        revealed = NUM_DIRT_TYPES * NUM_TOOLS if self.reveal_rule else 0
        # kind one-hot + colour one-hot + 10 agent features + held-tool colour
        _shape_obs = (
            (self.OBS_SIZE, self.OBS_SIZE,
             NUM_ITEM_CHANNELS + 10 + NUM_COLORS + revealed)
            if self.cnn
            else (self.OBS_SIZE**2 * (NUM_ITEM_CHANNELS + 10),)
        )

        return spaces.Box(
                low=0, high=1E9, shape=_shape_obs, dtype=jnp.uint8
            ), _shape_obs
    
    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        _shape = (
            (self.GRID_SIZE_ROW, self.GRID_SIZE_COL, NUM_TYPES + 4)
            if self.cnn
            else (self.GRID_SIZE_ROW* self.GRID_SIZE_COL * (NUM_TYPES + 4),)
        )
        return spaces.Box(low=0, high=1, shape=_shape, dtype=jnp.uint8)
    
    def render_tile(
        self,
        obj: int,
        agent_dir: Union[int, None] = None,
        agent_hat: bool = False,
        highlight: bool = False,
        tile_size: int = 32,
        subdivs: int = 3,
        terrain: Union[int, None] = None,
        beam_orient: Union[str, None] = None,
    ) -> onp.ndarray:
        """
        Render a tile and cache the result.

        terrain: the Items code of the terrain under an agent (river/dirt).
        The grid stores a single int per cell, so an agent stamp destroys
        the terrain value; the caller recovers it and passes it here so the
        agent triangle is drawn over the correct background.
        """

        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (agent_dir, agent_hat, highlight, tile_size, terrain, beam_orient)
        if obj:
            key = (obj, 0, 0, 0) + key if obj else key

        if key in self.tile_cache:
            return self.tile_cache[key]

        img = onp.full(
                shape=(tile_size * subdivs, tile_size * subdivs, 3),
                fill_value=SAND_COLOUR,
                dtype=onp.uint8,
            )

        if obj is not None and len(Items) <= obj < len(Items) + self.num_agents:
            # Draw the agent
            agent_color = self.PLAYER_COLOURS[obj-len(Items)]
            # Paint the terrain the agent is standing on first
            if terrain == Items.river or terrain == Items.potential_dirt:
                paint_water_tile(img)
            elif terrain is not None and bool(is_dirt(terrain)):
                paint_dirt_tile(img, terrain)
            elif terrain is not None and bool(is_tool(terrain)):
                paint_tool_tile(img, terrain)
            elif terrain == SOIL:
                paint_soil_tile(img)
        elif obj == Items.apple:
            paint_apple_tile(img)
        elif obj == APPLE_ON_SOIL:
            paint_soil_tile(img); paint_apple_tile(img)
        elif obj == Items.river or obj == Items.potential_dirt:
            paint_water_tile(img)
        elif obj is not None and bool(is_dirt(obj)):
            paint_dirt_tile(img, obj)
        elif obj is not None and bool(is_tool(obj)):
            paint_tool_tile(img, obj)
        elif obj == Items.wall:
            paint_wall_tile(img)
        elif obj == Items.interact:
            paint_beam_tile(img, (250, 238, 160), (243, 210, 60), beam_orient)
        elif obj == Items.clean_beam:
            paint_beam_tile(img, (200, 228, 250), (140, 198, 245), beam_orient)
        elif obj == 99:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (44.0, 160.0, 44.0))
        elif obj == 100:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (214.0, 39.0, 40.0))
        elif obj == 101:
            # white square
            fill_coords(img, point_in_rect(0, 1, 0, 1), (255.0, 255.0, 255.0))
        elif obj == SOIL:
            paint_soil_tile(img)

        # Overlay the agent on top
        if agent_dir is not None:
            draw_agent(img, agent_dir, agent_color)

        # Highlight the cell if needed (soft, so the obs windows don't wash
        # out the scene like the old 30% white blend did)
        if highlight:
            highlight_img(img, alpha=0.12)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        self.tile_cache[key] = img
        return img

    def render(
        self,
        state: State,
    ) -> onp.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        tile_size = 32
        highlight_mask = onp.zeros(onp.asarray(self.GRID).shape, dtype=bool)

        # Compute the total grid size
        width_px = self.GRID.shape[1] * tile_size
        height_px = self.GRID.shape[0] * tile_size

        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)

        # Pull device state to host ONCE; the per-cell loop below must stay
        # pure numpy/python (any jax comparison would be a blocking sync).
        grid = onp.asarray(state.grid)
        grid = onp.pad(
            grid, ((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)),
            constant_values=int(Items.wall)
        )
        agent_locs = onp.asarray(state.agent_locs)

        for a in range(self.num_agents):
            startx, starty = self.get_obs_point(state.agent_locs[a])
            startx, starty = int(startx), int(starty)
            highlight_mask[
                startx : startx + self.OBS_SIZE, starty : starty + self.OBS_SIZE
            ] = True

        # grid code -> facing direction, one entry per agent stamp
        first_agent_code = len(Items)
        agent_dirs = {
            first_agent_code + a: int(agent_locs[a, 2])
            for a in range(self.num_agents)
        }

        # grid code -> terrain under the agent. The agent stamp overwrote
        # the cell, but river cells are static (self._river_cells) and the
        # live waste map is tracked separately in state, so both survive.
        lay = self.layout_arrays(state); river_cells = lay["river_set"]
        # with random maps the empty orchard is drawn as soil, otherwise nothing would show where it is
        soil = lay["orchard_set"] if self.layout_pool is not None else set()
        waste_locs = onp.asarray(state.potential_dirt_and_dirt_locs)
        waste_labels = onp.asarray(state.potential_dirt_and_dirt_label)
        waste_map = {(int(r), int(c)): int(l)
                     for (r, c), l in zip(waste_locs, waste_labels)}
        agent_terrain = {}
        for a in range(self.num_agents):
            rc = (int(agent_locs[a, 0]), int(agent_locs[a, 1]))
            if rc in river_cells:
                agent_terrain[first_agent_code + a] = int(Items.river)
            elif rc in waste_map:
                agent_terrain[first_agent_code + a] = waste_map[rc]
            elif rc in soil:
                agent_terrain[first_agent_code + a] = SOIL

        # Render the grid
        beam_codes = (int(Items.interact), int(Items.clean_beam))
        for j in range(0, grid.shape[1]):
            for i in range(0, grid.shape[0]):
                cell = int(grid[i, j])
                if (i - self.PADDING, j - self.PADDING) in soil:      # orchard ground under empties and apples alike
                    cell = SOIL if cell == 0 else (APPLE_ON_SOIL if cell == int(Items.apple) else cell)
                beam_orient = None
                if cell in beam_codes:
                    h = ((j > 0 and grid[i, j - 1] == cell)
                         or (j + 1 < grid.shape[1] and grid[i, j + 1] == cell))
                    v = ((i > 0 and grid[i - 1, j] == cell)
                         or (i + 1 < grid.shape[0] and grid[i + 1, j] == cell))
                    if h and not v:
                        beam_orient = "h"
                    elif v and not h:
                        beam_orient = "v"
                tile_img = self.render_tile(
                    cell if cell != 0 else None,
                    agent_dir=agent_dirs.get(cell),
                    agent_hat=False,
                    highlight=bool(highlight_mask[i, j]),
                    tile_size=tile_size,
                    terrain=agent_terrain.get(cell),
                    beam_orient=beam_orient,
                )

                ymin = i * tile_size
                ymax = (i + 1) * tile_size
                xmin = j * tile_size
                xmax = (j + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img
        # Crop the padding; the image is NOT rotated any more, so it has the
        # same orientation as the map_ASCII the env was built from.
        img = img[
            (self.PADDING - 1) * tile_size : -(self.PADDING - 1) * tile_size,
            (self.PADDING - 1) * tile_size : -(self.PADDING - 1) * tile_size,
            :,
        ]
        return img

    def render_time(self, state, width_px) -> onp.array:
        inner_t = state.inner_t
        outer_t = state.outer_t
        tile_height = 32
        img = onp.zeros(shape=(2 * tile_height, width_px, 3), dtype=onp.uint8)
        tile_width = width_px // (self.num_inner_steps)
        j = 0
        for i in range(0, inner_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        tile_width = width_px // (self.num_outer_steps)
        j = 1
        for i in range(0, outer_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        return img
    
    def get_inequity_aversion_rewards_immediate(self, array, inner_t, target_agents=None, alpha=5, beta=0.05):
        """
        Calculate inequity aversion rewards using immediate rewards, based on equation (3) in the paper
        
        Args:
            array: shape: [num_agents, 1] immediate rewards r_i^t for each agent
            target_agents: list of agent indices to apply inequity aversion
            alpha: inequity aversion coefficient (when other agents' rewards are greater than self)
            beta: inequity aversion coefficient (when self's rewards are greater than others)
        Returns:
            subjective_rewards: adjusted subjective rewards u_i^t after inequity aversion
        """
        # Ensure correct input shape
        assert array.shape == (self.num_agents, 1), f"Expected shape ({self.num_agents}, 1), got {array.shape}"
        
        # Calculate inequality using immediate rewards
        r_i = array  # [num_agents, 1]
        r_j = jnp.transpose(array)  # [1, num_agents]
        
        # Calculate inequality
        disadvantageous = jnp.maximum(r_j - r_i, 0)  # when other agents' rewards are higher
        advantageous = jnp.maximum(r_i - r_j, 0)     # when self's rewards are higher
        
        # Create mask to exclude self-comparison
        mask = 1 - jnp.eye(self.num_agents)
        disadvantageous = disadvantageous * mask
        advantageous = advantageous * mask
        
        # Calculate inequality penalty
        n_others = self.num_agents - 1
        inequity_penalty = (alpha * jnp.sum(disadvantageous, axis=1, keepdims=True) +
                           beta * jnp.sum(advantageous, axis=1, keepdims=True)) / n_others

        # Calculate subjective rewards u_i^t = r_i^t - inequality penalty
        subjective_rewards = array - inequity_penalty

        subjective_rewards = jnp.where(jnp.all(array == 0), -(alpha + beta) * n_others, subjective_rewards)
        
        # Apply inequity aversion only to target agents if specified
        if target_agents is not None:
            target_agents_array = jnp.array(target_agents)
            agent_mask = jnp.zeros(self.num_agents, dtype=bool)
            agent_mask = agent_mask.at[target_agents_array].set(True)
            agent_mask = agent_mask.reshape(-1, 1)  # [num_agents, 1]
            return jnp.where(agent_mask, subjective_rewards, array),jnp.sum(disadvantageous, axis=1, keepdims=True),jnp.sum(advantageous, axis=1, keepdims=True)
        else:
            return subjective_rewards,jnp.sum(disadvantageous, axis=1, keepdims=True),jnp.sum(advantageous, axis=1, keepdims=True)

    def get_svo_rewards(self, array, w=0.5, ideal_angle_degrees=45, target_agents=None):
        """
        Reward shaping function based on Social Value Orientation (SVO)
        
        Args:
            array: shape: [num_agents, 1] immediate rewards r_i for each agent
            w: SVO weight to balance self-reward and social value (0 <= w <= 1)
               w=0 means completely selfish, w=1 means completely altruistic
            ideal_angle_degrees: ideal angle in degrees
               - 45 degrees means complete equality
               - 0 degrees means completely selfish
               - 90 degrees means completely altruistic
            target_agents: list of agent indices to apply SVO
        
        Returns:
            shaped_rewards: rewards adjusted by SVO
            theta: reward angle in radians
        """
        # Ensure correct input shape
        assert array.shape == (self.num_agents, 1), f"Expected shape ({self.num_agents}, 1), got {array.shape}"
        
        # Convert ideal angle from degrees to radians
        ideal_angle = (ideal_angle_degrees * jnp.pi) / 180.0
        
        # Calculate group average reward r_j (excluding self)
        mask = 1 - jnp.eye(self.num_agents)  # [num_agents, num_agents]
        # Modified: use matrix multiplication to calculate other agents' rewards
        others_rewards = jnp.matmul(mask, array)  # [num_agents, 1]
        mean_others = others_rewards / (self.num_agents - 1)  # divide by number of other agents
        
        # Calculate reward angle θ(R) = arctan(r_j / r_i)
        r_i = array  # [num_agents, 1]
        r_j = mean_others  # [num_agents, 1]
        theta = jnp.arctan2(r_j, r_i)
        
        # Calculate social value oriented utility
        # U(r_i, r_j) = r_i - w * |θ(R) - ideal_angle|
        angle_deviation = jnp.abs(theta - ideal_angle)
        svo_utility = r_i - self.num_agents * w * angle_deviation

        # Apply SVO only to target agents if specified
        if target_agents is not None:
            target_agents_array = jnp.array(target_agents)
            agent_mask = jnp.zeros(self.num_agents, dtype=bool)
            agent_mask = agent_mask.at[target_agents_array].set(True)
            agent_mask = agent_mask.reshape(-1, 1)  # [num_agents, 1]
            return jnp.where(agent_mask, svo_utility, array), theta
        else:
            return svo_utility, theta

    def get_standardized_svo_rewards(self, array, w=0.5, ideal_angle_degrees=45, target_agents=None):
        """
        Reward shaping function based on Social Value Orientation (SVO)
        """
        # Ensure correct input shape
        assert array.shape == (self.num_agents, 1), f"Expected shape ({self.num_agents}, 1), got {array.shape}"
        
        # Convert ideal angle from degrees to radians
        ideal_angle = (ideal_angle_degrees * jnp.pi) / 180.0
        
        # Calculate group average reward r_j (excluding self)
        mask = 1 - jnp.eye(self.num_agents)
        others_rewards = jnp.matmul(mask, array)
        mean_others = others_rewards / (self.num_agents - 1)
        
        # Calculate reward angle θ(R) = arctan(r_j / r_i)
        r_i = array
        r_j = mean_others
        theta = jnp.arctan2(r_j, r_i)
        
        # Convert angle to [0, 2π] range
        theta = (theta + 2 * jnp.pi) % (2 * jnp.pi)
        
        # Calculate angle deviation and normalize to [0, 1] range
        angle_deviation = jnp.abs(theta - ideal_angle)
        angle_deviation = jnp.minimum(angle_deviation, 2 * jnp.pi - angle_deviation)  # take minimum deviation
        normalized_deviation = angle_deviation / jnp.pi  # normalize to [0, 1]
        
        # Use multiplicative form of penalty instead of subtraction
        svo_utility = r_i * (1 - w * normalized_deviation)
        
        # Apply SVO only to target agents if specified
        if target_agents is not None:
            target_agents_array = jnp.array(target_agents)
            agent_mask = jnp.zeros(self.num_agents, dtype=bool)
            agent_mask = agent_mask.at[target_agents_array].set(True)
            agent_mask = agent_mask.reshape(-1, 1)
            return jnp.where(agent_mask, svo_utility, array), theta
        else:
            return svo_utility, theta

    def get_cf_regret(self, cf_rewards, actions):
        """
        Counterfactual regret of every agent.
        Args:
            cf_rewards: jnp.ndarray of shape [num_agents, num_actions], the counterfactual reward of every action of every agent
            actions: jnp.ndarray of shape [num_agents], the action every agent actually took
        Returns:
            cf_regret: jnp.ndarray of shape [num_agents], the counterfactual regret of every agent
        """
        # 1. the best counterfactual reward of every agent
        max_cf_reward = jnp.max(cf_rewards, axis=1)  # [num_agents]
        # 2. the counterfactual reward of the action actually taken
        actual_cf_reward = cf_rewards[jnp.arange(self.num_agents), actions]  # [num_agents]
        # 3. the regret
        cf_regret = max_cf_reward - actual_cf_reward
        return cf_regret

    def get_cf_regret_from_state(self, key, state, actions):
        """
        Counterfactual regret of every agent: enumerate each agent's actions with the other agents' actions fixed and step the environment for the reward.
        Args:
            key: jax.random.PRNGKey
            state: the current environment state
            actions: jnp.ndarray of shape [num_agents], the action every agent actually took
        Returns:
            cf_regret: jnp.ndarray of shape [num_agents], the counterfactual regret of every agent
        """
        num_agents = self.num_agents
        num_actions = self.num_actions

        def agent_cf_rewards(agent_id):
            def single_action_cf(a_cf):
                # the counterfactual joint action
                cf_actions = actions.at[agent_id].set(a_cf)
                # step the environment for its reward
                _, _, rewards, _, _ = self.step_env(key, state, cf_actions)
                return rewards[agent_id]
            # enumerate every action of this agent
            return jax.vmap(single_action_cf)(jnp.arange(num_actions))  # [num_actions]

        # cf_rewards of all agents at once
        cf_rewards = jax.vmap(agent_cf_rewards)(jnp.arange(num_agents))  # [num_agents, num_actions]
        # the regret
        cf_regret = self.get_cf_regret(cf_rewards, actions)
        return cf_regret

    def get_simple_cf_regret(self, rewards):
        """
        An approximate regret from the current reward array only.
        Args:
            rewards: jnp.ndarray of shape [num_agents, 1]
        Returns:
            regret: jnp.ndarray of shape [num_agents]
        """
        max_reward = jnp.max(rewards)  # the largest immediate reward among all agents
        regret = max_reward - rewards.squeeze()
        return regret

