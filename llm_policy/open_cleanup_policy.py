"""OpenCleanup LLM harness: raw actions with ledgers.

  * the user message is nine blocks: progress, rules ledger, social ledger, plan
    ledger, last step + events, recent steps, current state, actions available
    now, output request (text in open_cleanup_prompt.py)
  * the model answers with mode + goal + action; the goal is kept, checked and
    shown back by the harness (PlanLedger), never by the model's memory
  * everything about other agents comes from this agent's own view only
    (SocialLedger): fully decentralized
  * strict output: an invalid reply is a "stay" and is reported on the next step

The environment construction, the curriculum, the model providers and the rule
ledger (Evidence) also live in this file.
"""
from __future__ import annotations

import argparse, json, os, re, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as onp

from opensocialjax.environments.open_cleanup.open_cleanup import (
    ATTR_HUE_NAMES, ATTR_SHADE_NAMES, Actions, DIRT_BASE, Items, NUM_COLORS, NUM_DIRT_TYPES, NUM_SHADES, TOOL_BASE, attr_tables, craft_recipes)
from llm_policy import open_cleanup_prompt as P
from opensocialjax.environments.discovery_rules import NO_EFFECT, TYPE_TOOL_BASE

# which tool structure the current env uses: "shared" = 7 working tools (one light-clearing tool per hue, one mid
# and one dark tool shared by every hue); "per-colour" = 15 working tools, one per waste colour. Set by make_env().
TOOLS = "shared"

PICK = 8
ACTION_NAMES = {0: "north", 1: "south", 2: "east", 3: "west", 4: "turn left", 5: "turn right",
                6: "fire", 7: "stay", 8: "pick up"}
FACING = {0: "south", 1: "east", 2: "north", 3: "west"}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MODES = ["craft", "clean", "harvest", "explore"]
GOAL_TYPES = ["move_to", "fire", "pick_up", "wait"]
STALE_AFTER = 40

# One edit to the model's own notebook (see Notebook). Shared with open_harvest_policy.
NOTE_SCHEMA = {"type": "object",
               "properties": {"op": {"type": "string", "enum": ["none", "add", "replace", "delete"]},
                              "id": {"type": "integer"},
                              "text": {"type": "string", "maxLength": 200}},
               "required": ["op", "id", "text"], "additionalProperties": False}

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        # "why_this_move", not "reasoning": Claude Opus 5.5 refused some prompts outright (stop_reason refusal,
        # category reasoning_extraction) when asked for a field called "reasoning", and the same prompt with any other
        # name went through, 3/3 each (2026-09-23). The parsed reply still calls it "reasoning" for the analysis tools.
        "why_this_move": {"type": "string", "maxLength": 500},
        "mode": {"type": "string", "enum": MODES},
        "goal": {"type": "object",
                 "properties": {"type": {"type": "string", "enum": GOAL_TYPES},
                                "cell": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                                "why": {"type": "string", "maxLength": 80}},
                 "required": ["type", "cell", "why"], "additionalProperties": False},
        # one edit to the model's own notebook: add an entry, replace or delete entry [id], or none.
        # strict json_schema wants every property present, so id is 0 and text "" where they do not apply.
        "note": NOTE_SCHEMA,
        "action": {"type": "integer", "minimum": 0, "maximum": 8},
    },
    "required": ["why_this_move", "mode", "goal", "note", "action"],
    "additionalProperties": False,
}


def waste_label(code):
    return f"{ATTR_HUE_NAMES[code // NUM_SHADES][0].upper()}{code % NUM_SHADES}"


# ---------------------------------------------------------------- environment, actions, rule ledger, provider
# environment construction and the scripted curriculum
# model action id -> env action; 8 = pick up (only with explicit_pickup)
RAW_TO_ENV = {0: int(Actions.down), 1: int(Actions.up), 2: int(Actions.left), 3: int(Actions.right),
              4: int(Actions.turn_left), 5: int(Actions.turn_right), 6: int(Actions.zap_clean),
              7: int(Actions.stay), 8: int(Actions.pick_up)}


FIRE, STAY = 6, 7


STEP = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}        # facing -> (drow, dcol)




def _window(env, loc):
    r, c, d = (int(v) for v in loc); h = env.OBS_SIZE // 2
    r_lo, c_lo = r - h, c - h
    if d == 0:
        r_lo += h - 1
    elif d == 1:
        c_lo += h - 1
    elif d == 2:
        r_lo -= h - 1
    elif d == 3:
        c_lo -= h - 1
    return r_lo, r_lo + env.OBS_SIZE - 1, c_lo, c_lo + env.OBS_SIZE - 1


_WINDOW_CHECKED = set()


def view_window(env, loc):
    """The cells the environment itself shows an agent at loc = (row, col, facing): its OBS_SIZE x OBS_SIZE window pushed
    forward -- 9 cells ahead, 1 behind, 5 to each side for OBS_SIZE 11 -- as in the env's _get_obs_point.
    Returns inclusive (r_lo, r_hi, c_lo, c_hi). The first call per env checks the arithmetic against the env itself."""
    if id(env) not in _WINDOW_CHECKED:
        for d in range(4):
            x, y = (int(v) for v in env.get_obs_point(jnp.array([40, 40, d], dtype=jnp.int16)))
            want = (x - env.PADDING, x - env.PADDING + env.OBS_SIZE - 1, y - env.PADDING, y - env.PADDING + env.OBS_SIZE - 1)
            assert _window(env, (40, 40, d)) == want, f"view window for facing {d}: {_window(env, (40, 40, d))} != env {want}"
        _WINDOW_CHECKED.add(id(env))
    return _window(env, loc)


def colour_name(c):
    """A waste colour in prose: the label the map draws, glossed once. The raw code used to lead
    here ("11 (cyan-dark)") while the map said C2, so one sentence carried two notations."""
    c = int(c)
    return f"{waste_label(c)} ({ATTR_HUE_NAMES[c // NUM_SHADES]}-{ATTR_SHADE_NAMES[c % NUM_SHADES]})"


def hand_text(env, held, pickups):
    """The hand, named by the pick-ups that made it and by nothing else. Whether those pick-ups
    happen to be a recipe is exactly what 2.3 says you find out by firing, so this must not say:
    naming it "crafted" would hand over 7 of the 16 orders for free, without a shot."""
    picks = [int(x) for x in pickups if int(x) >= 0]
    return f"what your pick-ups {picks} gave you" if picks else "nothing yet (you have picked up nothing)"


class Evidence:
    """Which pickup orders this agent fired with and what the shots did. Only
    what it saw; the inference is the model's job. Kept for the whole episode."""

    def __init__(self, k=2):
        self.k = k
        self.given = {}          # tool class -> window, handed over by --reveal-orders
        self.found = {}          # tool class -> window identified by a shot
        self.tested = {}         # window -> set of tool classes it is NOT
        # what this agent actually WATCHED happen, per window: (colour, "cleared") or (colour, new colour).
        # found/tested above generalise to tool classes and drive --needed-hint and the fire preview;
        # this is the raw record, and it is the only thing render() shows, so the ledger never tells the
        # agent about a hue it has not met or claims an effect on waste it has not hit.
        self.watched = {}        # window -> dict((colour, outcome) -> times)
        self.window = ()

    @staticmethod
    def cls(c):
        """The working-tool class that acts on waste colour c: ("colour", c) with per-colour tools; otherwise
        ("hue", h) for a light colour and the shared "mid" / "dark" tool for the others."""
        c = int(c)
        if not 0 <= c < NUM_DIRT_TYPES:
            return None
        if TOOLS == "per-colour":
            return ("colour", c)
        s = c % NUM_SHADES
        return "dark" if s == NUM_SHADES - 1 else "mid" if s > 0 else ("hue", c // NUM_SHADES)

    @staticmethod
    def short(cls):
        """Kind name without codes: "green-dark", "green-light", "mid", "dark"."""
        if isinstance(cls, tuple) and cls[0] == "colour":
            c = cls[1]
            return f"{ATTR_HUE_NAMES[c // NUM_SHADES]}-{ATTR_SHADE_NAMES[c % NUM_SHADES]}"
        if isinstance(cls, tuple):
            return f"{ATTR_HUE_NAMES[cls[1]]}-light"
        return cls

    @staticmethod
    def name(cls):
        """Short label with the waste codes the tool acts on (code = hue*3 + shade)."""
        if isinstance(cls, tuple) and cls[0] == "colour":
            return f"{Evidence.short(cls)} (waste {cls[1]})"
        if isinstance(cls, tuple):
            return f"{ATTR_HUE_NAMES[cls[1]]}-light (waste {cls[1] * NUM_SHADES})"
        mids = "/".join(str(h * NUM_SHADES + 1) for h in range(NUM_DIRT_TYPES // NUM_SHADES))
        darks = "/".join(str(h * NUM_SHADES + 2) for h in range(NUM_DIRT_TYPES // NUM_SHADES))
        return f"mid (waste {mids})" if cls == "mid" else f"dark (waste {darks})"

    @staticmethod
    def label(cls):
        """Full description used for identified / given tools: what it does, on which waste codes."""
        H = NUM_DIRT_TYPES // NUM_SHADES
        if isinstance(cls, tuple) and cls[0] == "colour":
            c = cls[1]
            if c % NUM_SHADES == 0:
                return f"{Evidence.short(cls)} tool: CLEARS waste {c} ({Evidence.short(cls)}) to water, nothing else"
            return (f"{Evidence.short(cls)} tool: turns waste {c} ({Evidence.short(cls)}) one shade lighter "
                    f"(into {c - 1} ({ATTR_HUE_NAMES[c // NUM_SHADES]}-{ATTR_SHADE_NAMES[c % NUM_SHADES - 1]})), nothing else")
        if isinstance(cls, tuple):
            c = cls[1] * NUM_SHADES
            return f"{ATTR_HUE_NAMES[cls[1]]}-light clearing tool: CLEARS waste {c} ({ATTR_HUE_NAMES[cls[1]]}-light) to water"
        if cls == "mid":
            return ("mid lightening tool: turns ANY mid waste " + ", ".join(str(h * NUM_SHADES + 1) for h in range(H))
                    + " one shade lighter (into " + ", ".join(str(h * NUM_SHADES) for h in range(H)) + ")")
        return ("dark lightening tool: turns ANY dark waste " + ", ".join(str(h * NUM_SHADES + 2) for h in range(H))
                + " one shade lighter (into " + ", ".join(str(h * NUM_SHADES + 1) for h in range(H)) + ")")

    def reveal(self, given):
        self.given = dict(given); self.found.update(given)

    def record(self, o):
        true_w = tuple(int(p) for p in o["window"] if p >= 0)
        if o["shots"] == 0 or len(self.window) < self.k or true_w != tuple(self.window):
            self.window = true_w          # no shot, or the hand changed under the shots: nothing to attribute
            return
        w, hit = tuple(self.window), set()

        def claim(cls):
            if any(ww == w and cc != cls for cc, ww in self.found.items()):
                return
            if cls in self.found and self.found[cls] != w:
                return
            self.found[cls] = w; hit.add(cls)

        log = self.watched.setdefault(w, {})
        changed = set()
        for c, n in o["cleared_by_colour"].items():
            if int(n) > 0:
                log[(int(c), "cleared")] = log.get((int(c), "cleared"), 0) + int(n)
                changed.add(int(c))
                if int(c) % NUM_SHADES == 0:
                    claim(self.cls(c))
        for c, d in o["transformed"].items():
            for y, n in d.items():
                if int(n) > 0:
                    log[(int(c), int(y))] = log.get((int(c), int(y)), 0) + int(n)
                    changed.add(int(c))
                    if int(y) == int(c) - 1:
                        claim(self.cls(c))
        for c in o["shot_targets"]:
            if int(c) not in changed:
                log[(int(c), "nothing")] = log.get((int(c), "nothing"), 0) + 1
            cls = self.cls(c)
            if cls is not None and cls not in hit and self.found.get(cls) != w:
                self.tested.setdefault(w, set()).add(cls)

    def render(self):
        """What this agent has watched its own shots do, window by window. Observations only: the
        ledger joins a shot to the hand that fired it and remembers it for the episode, and stops
        there. Colours are named as the map draws them, with the hue and shade spelled out. Whether [2, 0] also works on a hue that has not turned up is the agent's to work out."""
        def phrase(w):
            bits, nothing = [], []
            for (c, out), n in sorted(self.watched[w].items(), key=lambda kv: (kv[0][1] == "nothing", kv[0][0])):
                times = "" if n == 1 else f" ({n}x)"
                if out == "cleared":
                    bits.append(f"cleared {colour_name(c)}{times}")
                elif out == "nothing":
                    nothing.append(f"{colour_name(c)}{times}")      # grouped, so a dud window reads as one clause
                else:
                    bits.append(f"turned {colour_name(c)} into {colour_name(out)}{times}")
            if nothing:
                bits.append("no effect on " + ", ".join(nothing))
            return "; ".join(bits)

        given = set(tuple(w) for w in self.given.values())
        told = [f"  {list(w)} -> " + "; ".join(self.label(c) for c, ww in self.found.items() if tuple(ww) == w)
                for w in sorted(given)]
        fired = [f"  {list(w)} -> " + phrase(w) for w in sorted(self.watched) if w not in given and self.watched[w]]
        head = ("The orders that make each working tool (given to you at the start, all true):\n"
                + "\n".join(told) + "\n\n") if told else ""
        if not fired:
            return head + "Orders you have fired with so far: none."
        return head + "Orders you have fired with so far (what you watched them do):\n" + "\n".join(fired)


class OpenAICompatProvider:
    """vLLM / OpenAI behind the chat-completions API."""

    def __init__(self, model, base_url, api_key="EMPTY", temperature=0.2, timeout=300.0):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=2)
        self.model, self.temperature = model, temperature
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}

    def complete(self, system, messages, max_tokens=400, schema=None):
        kwargs = {"max_tokens": max_tokens, "temperature": self.temperature}
        if schema is not None:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "action", "schema": schema, "strict": True}}
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": system}] + messages, **kwargs)
        self.usage["calls"] += 1; self.usage["seconds"] += time.time() - t0
        if resp.usage:
            self.usage["input_tokens"] += resp.usage.prompt_tokens or 0
            self.usage["output_tokens"] += resp.usage.completion_tokens or 0
        return resp.choices[0].message.content or ""


MINI_MAP = [
    "FFFHFFFFHFFFFHFF",
    "FFHFFFFHFFFFHFFF",
    "  P   P    P ~  ",
    "             ~  ",
    "   P       P ~  ",
    "             ~  ",
    "             ~  ",      # station row (tool_row=6): stations on even columns
    "                ",
    "    P      P    ",
    "BBBBBBBBBBBBBBBB",
    "BBBBBBBBBBBBBBBB",
    "BBBBBBBBBBBBBBBB",
]


def make_env(agents=3, inner=200, outer=4, spawn=0.1, skew=0.8, on_train=False, apple_threshold=None,
             tools="shared", layouts="fixed", problem=None):
    """The evaluated game: held-out recipes (or the training split), mini map,
    attribute-craft with 4 base tools and 2-pickup recipes, explicit pick-up, individual rewards (the only kind).
    apple_threshold: river dirt fraction below which apples can grow (env default 0.4); 1.0 = growth rate
    scales with cleanliness from the first cleared cell."""
    from opensocialjax.environments.held_out import held_out_env
    global TOOLS
    assert tools in ("shared", "per-colour"), tools
    TOOLS = tools
    extra = {} if apple_threshold is None else {"thresholdDepletion": float(apple_threshold)}
    extra["report_beam"] = True      # the rule ledger is built from the environment's own verdict on each shot
    if tools == "per-colour":
        extra["per_colour_tools"] = True
    if layouts == "random":      # a map drawn per meta-episode from the held-out (or training) 20 % of the layout pool
        from opensocialjax.environments.open_cleanup.layouts import layout_split
        extra["layout_pool"] = layout_split(on_train=on_train)
    if problem is not None:      # replay one seed's problem (see problem_of): its map, its hidden rule, its waste hue
        assert layouts == "random", "a pinned problem is a map of the random pool"
        extra["layout_pool"] = extra["layout_pool"].subset([problem["layout"]])
        extra["fixed_dom_type"] = problem["hue"]
    env = held_out_env(num_agents=agents, skewed_waste=True, waste_skew=skew, dirtSpawnProbability=spawn, **extra,
                       map_ASCII=MINI_MAP, tool_row=6, chain_depth=1, num_inner_steps=inner, num_outer_steps=outer,
                       on_train=on_train, dom_hue_per_episode=True,
                       attr_rules=True, craft_rules=True, craft_tools=4, craft_len=2, explicit_pickup=True)
    if problem is not None:      # a one-rule pool: every reset draws that rule
        i = problem["rule"]
        env.rule_pool = jax.tree.map(lambda a_: a_[i:i + 1], env.rule_pool)
    return env


def problem_of(seed, agents, inner, outer, spawn, skew, on_train=False, apple_threshold=None, tools="shared"):
    """The problem a meta-episode with this seed plays (random layouts): its map (index in the layout split), its hidden
    rule (index in the rule pool) and its waste hue, read off the same reset run_episode does. make_env(problem=...)
    then replays exactly that problem, while a different run key changes everything else (spawns, waste shades,
    respawns, apples)."""
    env = make_env(agents, inner, outer, spawn, skew, on_train, apple_threshold, tools, layouts="random")
    _, kr, _ = jax.random.split(jax.random.PRNGKey(seed), 3)            # run_episode's reset key
    st = env.reset(kr)[1]
    enc = onp.asarray(st.rule_encoding); rules = onp.asarray(env.rule_pool.rules)
    hits = onp.nonzero((rules == enc[None]).all(axis=tuple(range(1, rules.ndim))))[0]
    assert len(hits) >= 1, "the drawn rule is not in the pool"
    return {"seed": int(seed), "layout": int(st.layout_id), "rule": int(hits[0]), "hue": int(st.dom_type)}


def true_recipes(env, state):
    """tool class -> pickup window, for --reveal-orders."""
    hue_tools, shade_tools = (onp.asarray(x) for x in attr_tables(state))
    products, recipes, valid = (onp.asarray(x) for x in craft_recipes(state, env.craft_len))
    by_product = {int(p): tuple(int(x) for x in r) for p, r, v in zip(products, recipes, valid) if bool(v)}
    if TOOLS == "per-colour":
        return {("colour", c): by_product[TYPE_TOOL_BASE + c] for c in range(NUM_DIRT_TYPES) if TYPE_TOOL_BASE + c in by_product}
    given = {("hue", h): by_product[int(t)] for h, t in enumerate(hue_tools) if int(t) in by_product}
    for s, nm in ((1, "mid"), (2, "dark")):
        if int(shade_tools[s]) in by_product:
            given[nm] = by_product[int(shade_tools[s])]
    return given


def curriculum_start(env, state, mode):
    """Training-only easier starts. ``river_tool``: every agent just south of
    the waste, facing north, holding a working tool (agent 0 the dominant
    hue's light tool, agent 1 the mid tool, agent 2 the dark tool, cycling);
    ``station_tool``: same tools, normal spawn. ``half_craft``: no tool; each
    agent stands ON the station that completes its recipe with the recipe's
    first pickup already in the window, so one correct pick-up crafts it."""
    if mode == "none":
        return state
    if mode in ("half_craft", "first_station", "near_first"):
        return _half_craft_start(env, state, done_first=(mode == "half_craft"), offset=(mode == "near_first"))
    n = env.num_agents
    hue_tools, shade_tools = (onp.asarray(x) for x in attr_tables(state))
    products, recipes, valid = (onp.asarray(x) for x in craft_recipes(state, env.craft_len))
    by_product = {int(p): [int(x) for x in r] for p, r, v in zip(products, recipes, valid) if bool(v)}
    g = onp.asarray(state.grid); m = (g >= DIRT_BASE) & (g < DIRT_BASE + NUM_DIRT_TYPES)
    hues = ((g[m] - DIRT_BASE) // NUM_SHADES).tolist()
    dom = max(set(hues), key=hues.count)
    wanted = ([TYPE_TOOL_BASE + dom * NUM_SHADES + s for s in range(NUM_SHADES)] if TOOLS == "per-colour"
              else [int(hue_tools[dom]), int(shade_tools[1]), int(shade_tools[2])])
    held = onp.asarray(state.held_tool).copy(); picks = onp.asarray(state.pickups).copy()
    for i in range(n):
        tool = wanted[i % 3] if wanted[i % 3] in by_product else wanted[1]
        held[i] = tool
        picks[i] = onp.array(by_product[tool][-env.craft_len:], dtype=picks.dtype)
    state = state.replace(held_tool=jnp.array(held, dtype=state.held_tool.dtype),
                          pickups=jnp.array(picks, dtype=state.pickups.dtype),
                          crafted=jnp.array(held >= env.craft_tools))
    if mode == "river_tool":
        row = int(onp.asarray(state.potential_dirt_and_dirt_locs)[:, 0].max()) + 1
        cols = onp.linspace(1, env.GRID_SIZE_COL - 2, n + 2)[1:-1].round().astype(int)
        locs = onp.asarray(state.agent_locs).copy(); grid = onp.asarray(state.grid).copy()
        for i in range(n):
            grid[locs[i, 0], locs[i, 1]] = int(Items.empty)
        for i in range(n):
            locs[i] = [row, int(cols[i]), 2]                  # 2 = facing north
            grid[row, int(cols[i])] = int(onp.asarray(env._agents)[i])
        new_locs = jnp.array(locs, dtype=state.agent_locs.dtype)
        state = state.replace(agent_locs=new_locs, reborn_locs=new_locs,      # _step copies reborn_locs first
                              grid=jnp.array(grid, dtype=state.grid.dtype))
    return state


def _half_craft_start(env, state, done_first=True, offset=False):
    """done_first=True: on the completing station with the first pickup done (one pick-up crafts).
    done_first=False ("first_station"): on the recipe's FIRST station with an empty window --
    pick up, walk to a station of the second base tool, pick up again."""
    n = env.num_agents
    hue_tools, shade_tools = (onp.asarray(x) for x in attr_tables(state))
    products, recipes, valid = (onp.asarray(x) for x in craft_recipes(state, env.craft_len))
    by_product = {int(p): [int(x) for x in r] for p, r, v in zip(products, recipes, valid) if bool(v)}
    g = onp.asarray(state.grid); m = (g >= DIRT_BASE) & (g < DIRT_BASE + NUM_DIRT_TYPES)
    hues = ((g[m] - DIRT_BASE) // NUM_SHADES).tolist()
    dom = max(set(hues), key=hues.count)
    wanted = ([TYPE_TOOL_BASE + dom * NUM_SHADES + s for s in range(NUM_SHADES)] if TOOLS == "per-colour"
              else [int(hue_tools[dom]), int(shade_tools[1]), int(shade_tools[2])])
    lay = env.layout_arrays(state); tools = lay["tools"]; tgrid = lay["tool_grid"]
    held = onp.asarray(state.held_tool).copy(); picks = onp.asarray(state.pickups).copy()
    locs = onp.asarray(state.agent_locs).copy(); grid = onp.asarray(state.grid).copy()
    for i in range(n):
        grid[locs[i, 0], locs[i, 1]] = int(Items.empty)
    used = set()
    for i in range(n):
        tool = wanted[i % 3] if wanted[i % 3] in by_product else wanted[1]
        first, last = by_product[tool][-2], by_product[tool][-1]
        if done_first:
            held[i] = first; picks[i] = onp.array([-1, first], dtype=picks.dtype)     # first pickup done
            stand_on = last
        else:
            picks[i] = onp.array([-1, -1], dtype=picks.dtype)                         # nothing yet
            stand_on = first
        cands = [tuple(int(x) for x in rc) for rc in tools
                 if int(tgrid[rc[0], rc[1]]) == stand_on and tuple(int(x) for x in rc) not in used]
        if not cands:                                   # every station of that tool already taken: share one
            cands = [tuple(int(x) for x in rc) for rc in tools if int(tgrid[rc[0], rc[1]]) == stand_on]
        r, c = cands[i % len(cands)]; used.add((r, c))
        if offset:
            # "near_first": 1-3 cells away from that station (south of the station row,
            # sideways offset), so the agent must find and reach it first
            rng = onp.random.default_rng(int(onp.asarray(state.grid).sum()) + 7 * i)
            dist = int(rng.integers(1, 4)); dc = int(rng.integers(-dist, dist + 1)); dr = dist - abs(dc)
            r2, c2 = min(r + dr, env.GRID_SIZE_ROW - 1), min(max(c + dc, 0), env.GRID_SIZE_COL - 1)
            if int(grid[r2, c2]) in (int(Items.empty),):
                r, c = r2, c2
            else:
                r = min(r + dist, env.GRID_SIZE_ROW - 1)
        locs[i] = [r, c, 2]                                                        # facing north
        grid[r, c] = int(onp.asarray(env._agents)[i])
    new_locs = jnp.array(locs, dtype=state.agent_locs.dtype)
    return state.replace(held_tool=jnp.array(held, dtype=state.held_tool.dtype),
                         pickups=jnp.array(picks, dtype=state.pickups.dtype),
                         crafted=jnp.zeros(n, dtype=bool), agent_locs=new_locs, reborn_locs=new_locs,
                         grid=jnp.array(grid, dtype=state.grid.dtype))


def beam_cells(env, loc):
    """The four cells a beam covers: front, two ahead, front-right, front-left."""
    r, c, d = (int(x) for x in loc)
    f, rt, lt = STEP[d], STEP[(d + 1) % 4], STEP[(d - 1) % 4]
    cells = [(r + f[0], c + f[1]), (r + 2 * f[0], c + 2 * f[1]),
             (r + f[0] + rt[0], c + f[1] + rt[1]), (r + f[0] + lt[0], c + f[1] + lt[1])]
    return [(rr, cc) for rr, cc in cells if 0 <= rr < env.GRID_SIZE_ROW and 0 <= cc < env.GRID_SIZE_COL]


# ---------------------------------------------------------------- observation (U7) + facts
def observe(env, state, i):
    """Text of the NOW block and the facts the ledgers and ACTIONS NOW need."""
    grid = onp.asarray(state.grid); H, W = grid.shape
    locs = onp.asarray(state.agent_locs)
    r0, c0, d = (int(x) for x in locs[i])
    lay = env.layout_arrays(state)                          # the map this episode plays on
    soil = lay["orchard_set"] if env.layout_pool is not None else set()   # random maps: empty orchard cells are shown
    river_seen, orchard_seen = set(), set()
    held = int(onp.asarray(state.held_tool)[i])
    picks = [int(x) for x in onp.asarray(state.pickups)[i] if int(x) >= 0]
    pos_of = {(int(locs[j][0]), int(locs[j][1])): j for j in range(len(locs)) if j != i}

    waste, apples, others, water, rows, view, stations = {}, [], {}, 0, [], set(), {}
    r_lo, r_hi, c_lo, c_hi = view_window(env, (r0, c0, d))   # the env's own window: 9 ahead, 1 behind, 5 each side
    for r in range(r_lo, r_hi + 1):
        if not (0 <= r < H):
            continue                       # a row entirely outside the world: 3.2 already says those are ##
        cells = []
        for c in range(c_lo, c_hi + 1):
            if not (0 <= r < H and 0 <= c < W):
                cells.append("##"); continue
            view.add((r, c)); code = int(grid[r, c])
            if (r, c) == (r0, c0):
                cells.append("ME")
            elif (r, c) in pos_of:
                j = pos_of[(r, c)]; others[j] = (r, c, int(locs[j][2])); cells.append("@" + LETTERS[j])
            elif code == int(Items.wall):
                cells.append("##")
            elif code == int(Items.apple):
                cells.append("AP"); apples.append((r, c)); orchard_seen.add((r, c))
            elif code in (int(Items.river), int(Items.potential_dirt)):
                cells.append("~~"); water += 1; river_seen.add((r, c))
            elif DIRT_BASE <= code < DIRT_BASE + NUM_DIRT_TYPES:
                col = code - DIRT_BASE; cells.append(waste_label(col)); waste[(r, c)] = col; river_seen.add((r, c))
            elif TOOL_BASE <= code < TOOL_BASE + NUM_COLORS:
                cells.append(f"T{code - TOOL_BASE}"); stations.setdefault(code - TOOL_BASE, []).append((r, c))
            elif code >= len(Items):
                cells.append("@" + LETTERS[code - len(Items)] if code - len(Items) < len(LETTERS) else "@?")
            elif code == int(Items.clean_beam):
                cells.append("**")
            elif (r, c) in soil:
                cells.append(",,"); orchard_seen.add((r, c))
            else:
                cells.append("..")
        rows.append(f"r{r:>2} " + " ".join(cells))

    under = int(lay["tool_grid"][r0, c0])
    beam = [((r, c), int(grid[r, c]) - DIRT_BASE) for (r, c) in beam_cells(env, (r0, c0, d))
            if DIRT_BASE <= int(grid[r, c]) < DIRT_BASE + NUM_DIRT_TYPES]
    hue_counts = {}
    for col in waste.values():
        hue_counts[col // NUM_SHADES] = hue_counts.get(col // NUM_SHADES, 0) + 1

    lines = [f"You are agent {LETTERS[i]}, at ({r0}, {c0}), facing {FACING[d]}"
             + (f", STANDING ON station T{under}." if under >= 0 else ".")]
    window = (f"[{picks[0]}, {picks[1]}]" if len(picks) == 2 else f"[{picks[0]}, _]" if len(picks) == 1 else "[_, _]")
    nxt = (f"; your next pick-up x makes it [{picks[-1]}, x]" if picks else "; your next pick-up x makes it [x, _]")
    lines.append(f"You hold {hand_text(env, held, picks)}. Your window (last two pick-ups, oldest first): {window}{nxt}.")
    if waste:
        dom = max(hue_counts, key=hue_counts.get)
        by_shade = [sum(1 for col in waste.values() if col // NUM_SHADES == dom and col % NUM_SHADES == s) for s in range(NUM_SHADES)]
        n_w = len(waste)
        lines.append(f"In view: dominant hue {ATTR_HUE_NAMES[dom]} ("
                     + ", ".join(f"{ATTR_SHADE_NAMES[s]} {n}" for s, n in enumerate(by_shade))
                     # counts, not a percentage: a percentage of the visible cells read like the pollution of the
                     # whole river, and agents took it as the number apples depend on -- then "found" a different
                     # apple threshold every time they looked at a different stretch of it (2026-09-23)
                     + f"); river cells you can see: {n_w} waste, {water} water; apples {len(apples)}"
                     + ("; other agents: " + ", ".join(f"{LETTERS[j]} at ({r}, {c})" for j, (r, c, _) in sorted(others.items())) if others else "; no other agent")
                     + ".")
    else:
        lines.append(f"In view: {'no river' if water == 0 else 'river cells you can see: ' + str(water) + ' water, no waste'}; apples {len(apples)}"
                     + ("; other agents: " + ", ".join(f"{LETTERS[j]} at ({r}, {c})" for j, (r, c, _) in sorted(others.items())) if others else "; no other agent")
                     + ".")
    if under >= 0:                                     # the station under ME is hidden on the map
        stations.setdefault(under, []).append((r0, c0))
    lines.append("Stations in view: " + ("; ".join(
        f"T{t} at " + ", ".join(f"({r}, {c})" for r, c in sorted(v)) for t, v in sorted(stations.items())) if stations else "none") + ".")
    lines.append("Map:")
    lines.append("    " + " ".join(f"{c:>2}" if 0 <= c < W else "  " for c in range(c_lo, c_hi + 1)))
    lines.extend(rows)
    facts = {"pos": (r0, c0), "facing": d, "held": held, "picks": picks, "under": under, "beam": beam, "stations": stations,
             "others": others, "view": view, "waste": waste, "apples": set(apples), "water": water,
             "hue_counts": hue_counts, "tool_text": hand_text(env, held, picks),
             "river_seen": river_seen, "orchard_seen": orchard_seen if soil else set(), "waste_cells": lay["waste_cells"]}
    return "\n".join(lines), facts


def tool_class_of(evidence, picks):
    """Which known working tool the current window is, or None."""
    w = tuple(picks[-2:]) if len(picks) >= 2 else None
    for cls, win in evidence.found.items():
        if w is not None and tuple(win) == w:
            return cls
    return None


def effect_of(cls, code):
    """What a tool class does to a waste cell of colour code: 'clear', 'lighten' or None."""
    shade, hue = code % NUM_SHADES, code // NUM_SHADES
    if isinstance(cls, tuple) and cls[0] == "colour":
        return None if cls[1] != code else ("clear" if shade == 0 else "lighten")
    if cls == "dark" and shade == 2:
        return "lighten"
    if cls == "mid" and shade == 1:
        return "lighten"
    if isinstance(cls, tuple) and shade == 0 and cls[1] == hue:
        return "clear"
    return None


def needed_hint(evidence, facts):
    """RULES YOU KNOW addendum (--needed-hint): the waste in view counted by kind, each with the tool it needs
    (recipe from what is known: given or identified by a shot), and what the tool in your hand can change here."""
    counts = {}
    for code in facts["waste"].values():
        k = Evidence.cls(code)
        counts[k] = counts.get(k, 0) + 1
    order = [k for k in ("dark", "mid") if k in counts] + sorted((k for k in counts if isinstance(k, tuple)), key=lambda k: (-counts[k], str(k)))
    parts = []
    for k in order:
        n = counts.get(k, 0)
        if n == 0:
            continue
        w = evidence.found.get(k)
        kind = Evidence.short(k)
        parts.append(f"{n} {kind} (need the {kind} tool = {list(w) if w is not None else 'recipe unknown'})")
    lines = ["Waste in view: " + (", ".join(parts) + "." if parts else "none.")]
    cls = tool_class_of(evidence, facts["picks"])
    if cls is not None:
        kind = Evidence.short(cls)
        n = sum(1 for code in facts["waste"].values() if effect_of(cls, code))
        can = f"it can change the {n} {kind} cell{'s' if n != 1 else ''}, nothing else" if n else "it can change none of the cells in view"
        lines.append(f"Your hand {facts['picks'][-2:]} is the {kind} tool: {can}.")
    elif len(facts["picks"]) >= 2:
        lines.append(f"Your hand {facts['picks'][-2:]} is not a known working tool.")
    else:
        lines.append("Your hand is a base tool: it does nothing to waste.")
    return "\n".join(lines)


def actions_now(facts, evidence=None, hint=False):
    """The ACTIONS NOW block: what each action would do right now."""
    if hint and evidence is not None and facts["beam"]:
        cls = tool_class_of(evidence, facts["picks"])
        fire = "beam covers " + ", ".join(
            f"{waste_label(code)} at ({r}, {c})" + (f" -> would {effect_of(cls, code)}" if cls is not None and effect_of(cls, code) else " -> no effect")
            for (r, c), code in facts["beam"])
    else:
        fire = ("beam covers " + ", ".join(f"{waste_label(code)} at ({r}, {c})" for (r, c), code in facts["beam"])
                if facts["beam"] else "nothing in the beam")
    items = ["0 north", "1 south", "2 east", "3 west", "4 turn left", "5 turn right", f"6 fire ({fire})", "7 stay"]
    if facts["under"] >= 0:
        picks = facts["picks"]
        nxt = f"[{picks[-1]}, {facts['under']}]" if picks else f"[{facts['under']}, _]"
        items.append(f"8 pick up tool {facts['under']} here -> window becomes {nxt}"
                     + (f" (this replaces what {picks[-2:]} gave you)" if len(picks) >= 2 else ""))
    return " · ".join(items)


# ---------------------------------------------------------------- the model's own memory
class Notebook:
    """What the model has worked out about PLAYING, in its own words.

    The other ledgers in this file are the harness thinking for the model: Evidence records what each
    shot did, PlanLedger tracks the goal, SocialLedger remembers what was seen of the others. Those
    cover the facts. This one covers what no ledger can compute -- an order of doing things that paid
    off, a way of working that cost steps, what the others are up to -- and it thinks for nobody: it
    applies the edit the model asks for and shows the result back, never judging an entry's content.

    Every entry is numbered, and a number means the same entry for the whole trial: deleting [2] does
    not turn [3] into [2], so an edit can never land on the wrong line because the model counted from a
    stale view. Three kinds of entry, all editable: the lessons and the todo list the model wrote when
    the last trial ended, and the notes it has added since. Each step the model may add an entry,
    replace one, delete one, or leave the notebook alone. A lesson carried over from the last trial is
    as open to correction as anything written this trial -- those are the ones most likely to be
    wrong, and before 2026-09-23 a wrong one stayed on the page for the whole trial.

    At a trial boundary consolidate() asks the model to rewrite the whole thing; the answer replaces
    every entry and numbering starts again at 1, so something survives into the next trial only if the
    model carried it. Facts do not need carrying: Evidence keeps those for the whole episode.
    """

    OPS = ("none", "add", "replace", "delete")

    def __init__(self, max_notes=40, max_rules=12, max_open=4, summarise=None, summarise_schema=None):
        self.max_notes, self.max_rules, self.max_open = max_notes, max_rules, max_open
        # the trial-boundary question is game-specific (what is worth carrying differs); default: Clean Up's
        self.summarise = summarise or P.build_summarise
        self.summarise_schema = summarise_schema or P.SUMMARISE_SCHEMA
        self.entries = []               # dicts: id, kind ("lesson" / "todo" / "note"), text, t (step written), edited (step or None)
        self.next_id = 1
        self.written_after = None       # 0-based index of the trial the current lessons were written after
        self.dropped = 0                # notes pushed off the end by max_notes, reported so the loss is visible
        self.history = []               # one record per consolidation, for the decisions log
        self.edits = []                 # every replace / delete, with the text before, for the decisions log
        self.n_notes = 0                # every note ever added, for the run summary

    # read-only views, kept for the run summary and for callers of the old notebook
    lessons = property(lambda self: [e["text"] for e in self.entries if e["kind"] == "lesson"])
    todo = property(lambda self: [e["text"] for e in self.entries if e["kind"] == "todo"])
    notes = property(lambda self: [e for e in self.entries if e["kind"] == "note"])

    def _new(self, kind, text, t=None):
        e = {"id": self.next_id, "kind": kind, "text": text, "t": t, "edited": None}
        self.next_id += 1
        self.entries.append(e)
        return e

    def _find(self, i):
        return next((e for e in self.entries if e["id"] == i), None)

    def apply(self, edit, trial, t):
        """Carry out the model's edit. Returns a record of what happened: applied or not, why not, and for
        replace / delete the text that was there before, so the log shows every revision."""
        edit = edit or {}
        op = edit.get("op", "none") if isinstance(edit, dict) else "none"
        text = str(edit.get("text") or "").strip()[:200] if isinstance(edit, dict) else ""
        try:
            i = int(edit.get("id") or 0) if isinstance(edit, dict) else 0
        except (TypeError, ValueError):
            i = 0
        rec = {"op": op, "id": i, "text": text, "applied": False}
        if op not in self.OPS or op == "none":
            rec["why"] = "no edit" if op == "none" else f"unknown op {op!r}"
            return rec
        if op == "add":
            if not text:
                rec["why"] = "empty text"; return rec
            if any(e["text"] == text for e in self.entries):
                rec["why"] = "already in the notebook"; return rec
            e = self._new("note", text, t)
            self.n_notes += 1
            rec.update(applied=True, id=e["id"])
            while len(self.notes) > self.max_notes:
                oldest = self.notes[0]
                self.entries.remove(oldest)
                self.dropped += 1
            return rec
        e = self._find(i)
        if e is None:
            rec["why"] = f"there is no entry [{i}]"; return rec
        rec["before"] = e["text"]; rec["kind"] = e["kind"]
        if op == "delete":
            self.entries.remove(e)
        else:                           # replace
            if not text:
                rec["why"] = "replace needs the new text (use delete to remove an entry)"; return rec
            e["text"], e["edited"] = text, t
        rec["applied"] = True
        self.edits.append(dict(rec, trial=trial, t=t))
        return rec

    def render(self):
        """The YOUR NOTES block, every entry under its number."""
        def line(e):
            when = f"t{e['t']}: " if e["t"] is not None else ""
            fixed = f" (revised t{e['edited']})" if e["edited"] is not None else ""
            return f"  [{e['id']}] {when}{e['text']}{fixed}"
        parts = []
        groups = (("lesson", "What you worked out" + (f" by the end of trial {self.written_after + 1}"
                                                     if self.written_after is not None else "") + ":"),
                  ("todo", "What you meant to try this trial:"),
                  ("note", "Notes you have added since:"))
        for kind, head in groups:
            es = [e for e in self.entries if e["kind"] == kind]
            if es:
                parts.append(head + "\n" + "\n".join(line(e) for e in es))
        if self.dropped:
            parts.append(f"({self.dropped} older note(s) fell off the end of the notebook before you wrote them up.)")
        return "\n\n".join(parts) or "Empty. You have not written anything down yet."

    def consolidate(self, provider, system, trial, outer, max_tokens=800, outcome=""):
        """Trial boundary: show the model its notebook and let it rewrite it. The answer becomes the notebook.

        An unparsable answer leaves the notebook untouched, which is the safe failure: the model keeps what it
        had instead of losing a trial's worth of findings to a malformed reply. Returns a record for the log."""
        before = self.render()
        question = self.summarise(before, trial, outer, self.max_rules, self.max_open, outcome=outcome)
        raw = provider.complete(system, [{"role": "user", "content": question}], max_tokens, schema=self.summarise_schema)
        rec = {"trial": trial, "before": before, "raw": raw, "ok": False,
               "n_notes": len(self.notes), "dropped": self.dropped,
               "edits": sum(1 for x in self.edits if x["trial"] == trial)}
        obj = _loads_loose(raw)

        def lines(v, cap):
            """A list of non-empty strings, capped. Anything else (a bare string above all, which
            would iterate into single letters) is rejected, so a malformed answer cannot overwrite
            the notebook with rubbish."""
            if not isinstance(v, list):
                return None
            return [str(x).strip() for x in v if isinstance(x, (str, int, float)) and str(x).strip()][:cap]

        lessons = lines((obj or {}).get("lessons"), self.max_rules)
        todo = lines((obj or {}).get("todo") or [], self.max_open)
        if lessons is None or todo is None:
            rec["why"] = "not a list of lines"
            self.history.append(rec)
            return rec
        self.entries, self.next_id, self.dropped, self.written_after = [], 1, 0, trial
        for x in lessons:
            self._new("lesson", x)
        for x in todo:
            self._new("todo", x)
        rec.update(ok=True, lessons=lessons, todo=todo, after=self.render())
        self.history.append(rec)
        return rec


def _loads_loose(text):
    """json.loads, tolerating a fenced block or prose around the object. None if there is no object in there."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    m = re.search(r"\{.*\}", text, re.S)
    for candidate in (text, m.group(0) if m else None):
        if candidate is None:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


# ---------------------------------------------------------------- ledgers
class SocialLedger:
    """What this agent has seen of the others, this trial. Nothing else."""

    def __init__(self, me, n):
        self.me, self.n = me, n
        self.reset()

    def reset(self):
        self.seen = {}               # j -> dict(step, pos, facing, fired, count, fires, cleared, turned, no_effect, blank)
        self.others_cleared = 0

    @staticmethod
    def _new():
        return {"count": 0, "fires": 0, "cleared": {}, "turned": {}, "no_effect": 0, "blank": 0}

    def update(self, step, others, shots, cleared_by_others):
        """shots: {j: [(kind, colour, to_colour), ...]} for every agent seen firing this step, listing what its beam did
        to the waste cells in this agent's view (kind cleared / turned / none); an empty list = no waste under the beam."""
        for j, (r, c, f) in others.items():
            e = self.seen.setdefault(j, self._new())
            e.update(step=step, pos=(r, c), facing=f, fired=(j in shots)); e["count"] += 1
        for j, effects in shots.items():
            e = self.seen.setdefault(j, self._new())
            e["fires"] += 1
            if not effects:
                e["blank"] += 1
            elif all(k == "none" for k, _, _ in effects):
                e["no_effect"] += 1
            for kind, col, to in effects:
                if kind == "cleared":
                    e["cleared"][col] = e["cleared"].get(col, 0) + 1
                elif kind == "turned":
                    e["turned"][(col, to)] = e["turned"].get((col, to), 0) + 1
        self.others_cleared += cleared_by_others

    @staticmethod
    def shots_text(e):
        """What this agent saw another agent's shots do, over the trial so far."""
        if not e.get("fires"):
            return ""
        parts = [f"cleared {n} {colour_name(c)}" for c, n in sorted(e["cleared"].items())]
        parts += [f"turned {n} {colour_name(c)} into {colour_name(y)}" for (c, y), n in sorted(e["turned"].items())]
        if e["no_effect"]:
            parts.append(f"{e['no_effect']} shot(s) hit waste with no effect")
        if e["blank"]:
            parts.append(f"{e['blank']} shot(s) with no waste under the beam in your view")
        return f"you saw it fire {e['fires']} time(s): " + "; ".join(parts)

    def summary(self):
        """One line for EARLIER TRIALS: what this agent saw of the others over the whole trial."""
        parts = []
        for j in range(self.n):
            if j == self.me:
                continue
            e = self.seen.get(j)
            parts.append(f"{LETTERS[j]} never seen" if not e else f"{LETTERS[j]} seen {e['count']} steps"
                         + (f", {self.shots_text(e)}" if e["fires"] else ", never seen firing"))
        return "other agents: " + "; ".join(parts) + f"; {self.others_cleared} river cells in your view were cleared by others."

    def render(self, step):
        lines = []
        for j in range(self.n):
            if j == self.me:
                continue
            e = self.seen.get(j)
            if not e:
                lines.append(f"Agent {LETTERS[j]}: not seen yet this trial.")
            else:
                # an agent can be on record only for its shots: in view on the step it fired, out of view once the
                # step was done (this agent turned or walked away). It then has no position or step of its own, and
                # asking for one crashed the whole run on its first appearance in a prompt.
                if "pos" in e:
                    ago = step - e["step"]
                    where = (f"last seen {'now' if ago == 0 else f'{ago} steps ago'} at {e['pos']} facing {FACING[e['facing']]}"
                             + (", firing" if e.get("fired") else "") + f"; seen {e['count']} times this trial")
                else:
                    where = "seen only while firing"
                lines.append(f"Agent {LETTERS[j]}: {where}" + (f"; {self.shots_text(e)}" if e["fires"] else "") + ".")
        lines.append(f"River cells cleared in your view but not by you this trial: {self.others_cleared}.")
        return "\n".join(lines)


class PlanLedger:
    """The model's goals, kept and checked by the harness."""
    KEEP_CLOSED = 5              # closed goals shown (harvest raises it)

    def __init__(self):
        self.reset()

    def reset(self):
        self.current = None
        self.closed = []             # dicts with status done/abandoned/stale/interrupted
        self.n_goals = 0
        self.consistent = self.inconsistent = 0
        self.changes = self.changes_without_event = 0

    @staticmethod
    def same(a, b):
        return a and b and a["type"] == b["type"] and tuple(a["cell"]) == tuple(b["cell"])

    def _close(self, status, step, why=""):
        if self.current:
            self.current.update(status=status, end_step=step, end_why=why)
            self.closed.append(self.current); self.closed = self.closed[-self.KEEP_CLOSED:]
        self.current = None

    def on_decision(self, goal, step, pos, action, had_event, station_cells=()):
        """Called with the parsed goal and the chosen action, before the env step.
        Returns (consistent, changed_without_event)."""
        if goal is None:                                  # invalid reply: keep the current goal
            return None, False
        goal = dict(goal, no_station=(goal["type"] == "pick_up" and tuple(goal["cell"]) not in set(station_cells)))
        changed_without_event = False
        if self.current and not self.same(self.current, goal):
            self.changes += 1
            if not had_event:
                self.changes_without_event += 1; changed_without_event = True
            self._close("abandoned", step, goal.get("why", ""))
        if self.current is None:
            self.n_goals += 1
            self.current = dict(goal, set_step=step, progress=0, start_dist=self.dist(goal, pos), idx=self.n_goals)
        self.current["progress"] += 1
        self.current["dist"] = self.dist(goal, pos)
        ok = self.is_consistent(goal, pos, action)
        self.consistent += ok; self.inconsistent += (not ok)
        return ok, changed_without_event

    def after_step(self, step, pos_after, fired, picked):
        """Close the current goal if the step completed it (or it went stale)."""
        g = self.current
        if not g:
            return
        done = ((g["type"] == "move_to" and tuple(pos_after) == tuple(g["cell"]))
                or (g["type"] == "fire" and fired) or (g["type"] == "pick_up" and picked)
                or g["type"] == "wait")
        if done:
            self._close("done", step)
        elif g["progress"] >= STALE_AFTER:
            self._close("stale", step)

    @staticmethod
    def dist(goal, pos):
        return abs(int(goal["cell"][0]) - pos[0]) + abs(int(goal["cell"][1]) - pos[1])

    @staticmethod
    def is_consistent(goal, pos, action):
        t = goal["type"]
        if t == "move_to":
            if tuple(goal["cell"]) == tuple(pos):
                return True
            dr = int(goal["cell"][0]) - pos[0]; dc = int(goal["cell"][1]) - pos[1]
            return ((action == 0 and dr < 0) or (action == 1 and dr > 0) or (action == 2 and dc > 0) or (action == 3 and dc < 0))
        if t == "fire":
            return action in (FIRE, 4, 5) or PlanLedger.dist(goal, pos) > 0 and action in (0, 1, 2, 3)
        if t == "pick_up":
            return action == PICK or (PlanLedger.dist(goal, pos) > 0 and action in (0, 1, 2, 3))
        return action == STAY

    def render(self, step):
        lines = []
        g = self.current
        if g:
            ago = step - g["set_step"]
            lines.append(f"Current goal #{g['idx']} (set step {g['set_step']}, {ago} steps ago): {g['type']} {tuple(int(x) for x in g['cell'])} — \"{g['why']}\"."
                         + (f" Distance now {g['dist']} (was {g['start_dist']})." if g["type"] == "move_to" else "")
                         + (" WARNING: there is no station at that cell in your view." if g.get("no_station") else ""))
        else:
            lines.append("No current goal (set one below).")
        if self.closed:
            lines.append("Earlier goals this trial: " + " · ".join(
                f"#{c['idx']} {c['type']} {tuple(int(x) for x in c['cell'])} — {c['status']} at step {c['end_step']}"
                + (f" (\"{c['end_why']}\")" if c["status"] == "abandoned" and c.get("end_why") else "") for c in self.closed))
        return "\n".join(lines)


# ---------------------------------------------------------------- the agent
def parse_reply(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            obj = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            obj = None
    if not isinstance(obj, dict):
        return None
    try:
        action = int(obj["action"]); g = obj["goal"]
        cell = [int(g["cell"][0]), int(g["cell"][1])]
        goal = {"type": str(g["type"]), "cell": cell, "why": str(g.get("why", ""))[:80]}
        mode = str(obj["mode"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if not 0 <= action <= 8 or goal["type"] not in GOAL_TYPES or mode not in MODES:
        return None
    why = obj.get("why_this_move", obj.get("reasoning", ""))          # older replies used "reasoning"
    return {"reasoning": str(why or "")[:500], "mode": mode, "goal": goal, "action": action,
            "note": parse_note(obj.get("note"))}


def parse_note(n):
    """The note field of a reply, normalised to {op, id, text}. A missing or malformed note is never a reason to
    reject the move: it becomes "none". A bare string (an older reply shape) is read as an add. Shared with harvest."""
    if isinstance(n, dict):
        try:
            nid = int(n.get("id") or 0)
        except (TypeError, ValueError):
            nid = 0
        return {"op": str(n.get("op") or "none"), "id": nid, "text": str(n.get("text") or "")[:200].strip()}
    if isinstance(n, str) and n.strip():
        return {"op": "add", "id": 0, "text": n.strip()[:200]}
    return {"op": "none", "id": 0, "text": ""}


def describe_outcome(o, pos_before):
    """LAST STEP: what the chosen action did."""
    a = o["action"]
    if o.get("invalid"):
        return "your reply was not valid JSON -> you stayed"
    if o.get("unavailable") == PICK:
        return "you chose 8 (pick up) -> NOT available: you are not standing on a station (see ACTIONS NOW) -> you stayed"
    if a in (0, 1, 2, 3):
        if tuple(o["moved_to"]) == tuple(pos_before):
            return f"you chose {a} ({ACTION_NAMES[a]}) -> blocked, you did not move"
        return f"you chose {a} ({ACTION_NAMES[a]}) -> moved {tuple(pos_before)} -> {tuple(o['moved_to'])}"
    if a in (4, 5):
        return f"you chose {a} ({ACTION_NAMES[a]}) -> now facing {o['facing_after']}"
    if a == FIRE:
        eff = [f"cleared {n} {colour_name(c)}" for c, n in o["cleared_by_colour"].items() if n]
        eff += [f"turned {n} {colour_name(c)} into {colour_name(y)}" for c, d in o["transformed"].items() for y, n in d.items() if n]
        if not o["shot_targets"]:
            return "you chose 6 (fire) -> nothing in the beam, the shot hit no waste"
        hit = ", ".join(waste_label(c) for c in o["shot_targets"])
        return f"you chose 6 (fire) -> hit {hit}: " + (", ".join(eff) if eff else "no effect")
    if a == PICK:
        if o["picked"]:
            return f"you chose 8 (pick up) -> picked up tool {o['picked_tool']}; you now hold {o['tool_text']}, last pick-ups {o['window_clean']}"
        return "you chose 8 (pick up) -> not on a station, nothing happened"
    return "you chose 7 (stay) -> nothing happened"


def recent_line(o, t, pos_before):
    a = o["action"]
    if o.get("invalid"):
        return f"t{t} invalid reply -> stay"
    if o.get("unavailable") == PICK:
        return f"t{t} pick up (not on a station) -> stay"
    if a in (0, 1, 2, 3):
        return f"t{t} {ACTION_NAMES[a]} -> {tuple(o['moved_to'])}" + (" (blocked)" if tuple(o["moved_to"]) == tuple(pos_before) else "")
    if a in (4, 5):
        return f"t{t} {ACTION_NAMES[a]} -> facing {o['facing_after']}"
    if a == FIRE:
        if not o["shot_targets"]:
            return f"t{t} fire -> nothing in the beam"
        eff = (f"cleared {o['cleared']}" if o["cleared"] else
               ("lightened " + str(sum(sum(d.values()) for d in o["transformed"].values())) if o["transformed"] else "no effect"))
        return f"t{t} fire -> hit {','.join(waste_label(c) for c in o['shot_targets'])}: {eff}"
    if a == PICK:
        return f"t{t} pick up -> " + (f"tool {o['picked_tool']}, window {o['window_clean']}" if o["picked"] else "not on a station")
    return f"t{t} stay"


def beam_verdicts(info, n):
    """agent -> {(row, col): (kind, colour, new colour)} for every waste cell that agent's beam covered this step,
    read from the environment's own verdicts (report_beam=True). kind is "cleared", "turned" or "none". A cell the
    footprint names twice (a side target at the map edge falls back to the front cell) is counted once."""
    rc, hit = onp.asarray(info["beam_rc"]), onp.asarray(info["beam_hit"])
    before, after = onp.asarray(info["beam_before"]), onp.asarray(info["beam_after"])
    out = {}
    for j in range(n):
        eff = {}
        for k in range(rc.shape[1]):
            b = int(before[j, k])
            if not hit[j, k] or not (DIRT_BASE <= b < DIRT_BASE + NUM_DIRT_TYPES):
                continue
            cell, col, a = (int(rc[j, k, 0]), int(rc[j, k, 1])), b - DIRT_BASE, int(after[j, k])
            if a == int(NO_EFFECT):
                eff[cell] = ("none", col, None)
            elif a in (int(Items.river), int(Items.potential_dirt)):
                eff[cell] = ("cleared", col, None)
            else:
                eff[cell] = ("turned", col, a - DIRT_BASE)
        out[j] = eff
    return out


class Agent:
    def __init__(self, provider, system, me, n_agents, history=5, max_tokens=800, strict=True, hint=False, random_layout=False):
        self.provider, self.system, self.me = provider, system, me
        self.random_layout = random_layout
        self.history_len, self.max_tokens, self.strict, self.hint = history, max_tokens, strict, hint
        # Evidence works out what each shot proved and shows it under RULES YOU KNOW; the notebook holds what
        # the model works out for itself, which is everything the harness cannot: how to play this thing.
        self.notebook = Notebook()
        self.evidence = Evidence()
        self.social = SocialLedger(me, n_agents)
        self.plan = PlanLedger()
        self.recent = deque(maxlen=history)
        self.last_text, self.events = "", []
        self.parse_failures = 0
        self.unavailable = 0
        self.mode_counts = {m: 0 for m in MODES}
        # kept across trials: one summary line per finished trial (own results + what was seen of the others),
        # and every station this agent has had in view (the layout is the same every trial)
        self.earlier = []
        self.stations_seen = {}
        self.river_seen, self.orchard_seen = set(), set()      # random maps: where the river and the orchard were seen
        self.trial_no = 0
        self._new_trial_stats()

    def _new_trial_stats(self):
        self.tstats = {"cleared": 0, "apples": 0.0, "fires": 0, "river": 0, "first_tool": None}

    def trial_summary(self):
        ts = self.tstats
        own = (f"Trial {self.trial_no + 1}: you cleared {ts['cleared']} waste cells, ate {ts['apples']:.0f} apples, fired {ts['fires']} times, "
               f"spent {ts['river']} steps {'in or next to the river' if self.random_layout else 'in the river rows'}; "
               + (f"first working tool in hand at step {ts['first_tool']}." if ts["first_tool"] is not None else "never held a working tool."))
        return own + "\n" + f"Trial {self.trial_no + 1}, " + self.social.summary()

    def consolidate(self, trial, outer):
        """Ask the model to rewrite its notes now that a trial has ended. Call this BEFORE new_trial(),
        while the trial being summarised is still the current one -- trial_summary() reads self.tstats,
        and the model needs it: it is being asked what paid off, so it has to see what the trial got."""
        return self.notebook.consolidate(self.provider, self.system, trial, outer, self.max_tokens,
                                         outcome=self.trial_summary())

    def new_trial(self):
        self.earlier.append(self.trial_summary())
        self.trial_no += 1; self._new_trial_stats()
        self.social.reset(); self.plan.reset(); self.recent.clear()
        self.last_text = ""; self.events = ["NEW TRIAL: the map was re-laid (same layout), the river is polluted again and your hand is empty. "
                                            "RULES YOU KNOW and EARLIER TRIALS are kept, and so is what you just wrote down; the step-by-step notes behind it are gone."]

    def stations_line(self, facts):
        for tool, cells in facts["stations"].items():
            self.stations_seen.setdefault(tool, set()).update(cells)
        lines = []
        if self.stations_seen:
            lines.append("Stations you have seen (same places every trial): " + "; ".join(
                f"T{t} at " + ", ".join(f"({r}, {c})" for r, c in sorted(v)) for t, v in sorted(self.stations_seen.items())) + ".")
        if self.random_layout:
            self.river_seen |= facts["river_seen"]; self.orchard_seen |= facts["orchard_seen"]
            def box(cells):
                rs = [r for r, _ in cells]; cs = [c for _, c in cells]
                return f"rows {min(rs)}-{max(rs)}, cols {min(cs)}-{max(cs)} ({len(cells)} cells seen)"
            lines.append("River seen (same place every trial): " + (box(self.river_seen) if self.river_seen else "not yet")
                         + "; orchard seen: " + (box(self.orchard_seen) if self.orchard_seen else "not yet") + ".")
        return "\n".join(lines)

    def decide(self, obs_text, facts, trial, outer, t, inner, cleared, apples):
        progress = f"Trial {trial + 1} of {outer}, step {t} of {inner}. This trial: you cleared {cleared} waste cells, you ate {apples:.0f} apples."
        last = "\n".join(([self.last_text] if self.last_text else []) + (["Events: " + " ".join(self.events)] if self.events else []))
        rules = self.evidence.render() + ("\n" + needed_hint(self.evidence, facts) if self.hint else "")
        notes = self.notebook.render()
        seen = self.stations_line(facts)
        if seen:                                           # right after "Stations in view: ..." in NOW
            obs_text = re.sub(r"(Stations in view: [^\n]*\n)", lambda m: m.group(1) + seen + "\n", obs_text, count=1)
        content = P.compose_user(progress, rules, self.social.render(t), self.plan.render(t),
                                 last, "\n".join(self.recent), obs_text, actions_now(facts, self.evidence, self.hint),
                                 earlier="\n".join(self.earlier), notes=notes)
        messages = [{"role": "user", "content": content}]
        raw = self.provider.complete(self.system, messages, self.max_tokens, schema=ACTION_SCHEMA)
        reply = parse_reply(raw)
        if reply is None and not self.strict:
            retry = messages + [{"role": "assistant", "content": raw},
                                {"role": "user", "content": "That was not a valid reply. Reply with ONLY the JSON object."}]
            raw2 = self.provider.complete(self.system, retry, self.max_tokens, schema=ACTION_SCHEMA)
            reply = parse_reply(raw2); raw = raw + "\n---RETRY---\n" + raw2
        think = getattr(self.provider, "last_reasoning", "")
        if think:
            raw = raw + "\n---THINK---\n" + think
        had_event = bool(self.events)
        if reply is None:
            self.parse_failures += 1
            reply = {"reasoning": "(invalid)", "mode": None, "goal": None, "action": STAY, "invalid": True}
        elif reply["action"] == PICK and facts["under"] < 0:
            # not in ACTIONS NOW: the step is a stay, and LAST STEP says why
            self.unavailable += 1
            self.mode_counts[reply["mode"]] += 1
            reply = dict(reply, action=STAY, unavailable=PICK)
        else:
            self.mode_counts[reply["mode"]] += 1
        if not reply.get("invalid"):
            res = self.notebook.apply(reply.get("note"), trial, t)
            reply["note_result"] = res
            reply["note_kept"] = bool(res["applied"] and res["op"] == "add")
            if res["op"] != "none" and not res["applied"]:
                # an edit that did not take is said so on the next step, or the model believes it did
                self.note_feedback = f"Your note edit ({res['op']} [{res['id']}]) was not applied: {res['why']}."
        reply["gen_tokens"] = getattr(self.provider, "last_tokens", None)        # completion tokens incl. thinking
        reply["gen_seconds"] = getattr(self.provider, "last_seconds", None)
        station_cells = [rc for v in facts["stations"].values() for rc in v]
        ok, chg = self.plan.on_decision(reply["goal"], t, facts["pos"], reply.get("unavailable") or reply["action"], had_event, station_cells)
        reply["consistent"], reply["changed_without_event"] = ok, chg
        return reply, content, raw

    def after_step(self, t, o, facts_before, facts_after, cleared_by_others, fired_in_view, trial_end, apples=0.0):
        self.evidence.record(o)
        ts = self.tstats
        ts["cleared"] += o["cleared"]; ts["apples"] += apples; ts["fires"] += o["shots"]
        mr, mc = o["moved_to"]; wc = facts_before["waste_cells"]                 # in or next to the river (fixed map: rows 0-2)
        ts["river"] += int(bool((onp.abs(wc[:, 0] - mr) + onp.abs(wc[:, 1] - mc)).min() <= 1))
        if ts["first_tool"] is None and facts_after["held"] >= 4:
            ts["first_tool"] = t
        self.last_text = f"Last step ({t}): " + describe_outcome(o, facts_before["pos"])
        self.recent.append(recent_line(o, t, facts_before["pos"]))
        self.plan.after_step(t, o["moved_to"], o["shots"] > 0, o["picked"])
        self.social.update(t + 1, facts_after["others"], fired_in_view, cleared_by_others)
        ev = []
        came = sorted(set(facts_after["others"]) - set(facts_before["others"]))
        left = sorted(set(facts_before["others"]) - set(facts_after["others"]))
        if came:
            ev.append("Agent " + ", ".join(f"{LETTERS[j]} came into your view at {facts_after['others'][j][:2]}" for j in came) + ".")
        if left:
            ev.append("Agent " + ", ".join(LETTERS[j] for j in left) + " left your view.")
        for j, effects in sorted(fired_in_view.items()):
            if not effects:
                ev.append(f"Agent {LETTERS[j]} fired; its beam covered no waste in your view.")
                continue
            hit = ", ".join(waste_label(c) for _, c, _ in effects)
            done = [f"cleared {sum(1 for k, c2, _ in effects if k == 'cleared' and c2 == c)} {colour_name(c)}"
                    for c in sorted({c for k, c, _ in effects if k == "cleared"})]
            done += [f"turned {sum(1 for k, c2, y2 in effects if k == 'turned' and (c2, y2) == (c, y))} {colour_name(c)} into {colour_name(y)}"
                     for c, y in sorted({(c, y) for k, c, y in effects if k == "turned"})]
            ev.append(f"Agent {LETTERS[j]} fired at {hit}: " + (", ".join(done) if done else "no effect") + ".")
        unattributed = cleared_by_others - sum(1 for effs in fired_in_view.values() for k, _, _ in effs if k == "cleared")
        if unattributed > 0:
            ev.append(f"{unattributed} river cell(s) in your view were cleared by someone you did not see firing.")
        if len(facts_after["apples"]) > len(facts_before["apples"]):
            ev.append("An apple appeared in your view.")
        if facts_after["tool_text"] != facts_before["tool_text"] and not o["picked"]:
            ev.append(f"Your hand changed: you now hold {facts_after['tool_text']}.")
        if o.get("bumps3"):
            ev.append("You have been blocked three steps in a row.")
        if getattr(self, "note_feedback", None):
            ev.append(self.note_feedback); self.note_feedback = None
        if trial_end:
            ev = []                                        # new_trial() sets the reset event
        self.events = ev


# ---------------------------------------------------------------- scripted stand-in
class ChatProvider(OpenAICompatProvider):
    """OpenAICompatProvider plus extra_body, e.g. chat_template_kwargs={"enable_thinking": False} for Qwen3.5
    (its template thinks by default; the 30B Instruct comparison model never thinks)."""

    def __init__(self, *args, extra_body=None, reasoning_effort=None, think_budget=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_body = extra_body or {}
        # budgeted thinking (models whose thinking has no length control, e.g. Qwen3.5-4B): pass 1 thinks for at
        # most `think_budget` tokens; pass 2 closes the think block and answers with the JSON schema enforced.
        self.think_budget = think_budget
        # OpenAI reasoning models (gpt-5*, gpt-6*, o*) reject max_tokens/temperature and take
        # max_completion_tokens (+ reasoning_effort); their thinking counts against the budget.
        self.reasoning = bool(reasoning_effort) or self.model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
        self.reasoning_effort = reasoning_effort

    def complete(self, system, messages, max_tokens=400, schema=None):
        if self.think_budget:
            return self._complete_budgeted(system, messages, max_tokens, schema)
        if self.reasoning:
            kwargs = {"max_completion_tokens": max_tokens}
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs = {"max_tokens": max_tokens, "temperature": self.temperature}
        if schema is not None:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "action", "schema": schema, "strict": True}}
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": system}] + messages, **kwargs)
        self.usage["calls"] += 1; self.usage["seconds"] += time.time() - t0
        self.last_reasoning_tokens = 0
        if resp.usage:
            self.usage["input_tokens"] += resp.usage.prompt_tokens or 0
            self.usage["output_tokens"] += resp.usage.completion_tokens or 0
            det = getattr(resp.usage, "completion_tokens_details", None)          # OpenAI: reasoning tokens are inside completion_tokens
            self.last_reasoning_tokens = int(getattr(det, "reasoning_tokens", 0) or 0) if det else 0
            self.usage["reasoning_tokens"] = self.usage.get("reasoning_tokens", 0) + self.last_reasoning_tokens
            pdet = getattr(resp.usage, "prompt_tokens_details", None)             # and cached prefix tokens inside prompt_tokens
            self.usage["cached_tokens"] = self.usage.get("cached_tokens", 0) + int(getattr(pdet, "cached_tokens", 0) or 0) if pdet else self.usage.get("cached_tokens", 0)
        msg = resp.choices[0].message
        # thinking mode (vLLM --reasoning-parser): the field is `reasoning_content` or, in newer servers, `reasoning`
        self.last_reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        self.last_tokens = int(resp.usage.completion_tokens or 0) if resp.usage else 0
        self.last_seconds = round(time.time() - t0, 2)
        return msg.content or ""


    CLOSE_THINK = "\n\nConsidering the limited time, I will stop thinking here and answer now based on the thinking so far."

    def _complete_budgeted(self, system, messages, max_tokens, schema):
        full = [{"role": "system", "content": system}] + messages
        tk = dict(self.extra_body.get("chat_template_kwargs", {})); tk["enable_thinking"] = True
        sampling = {k: v for k, v in self.extra_body.items() if k != "chat_template_kwargs"}   # top_p / top_k / presence_penalty
        t0 = time.time()
        r1 = self.client.chat.completions.create(model=self.model, messages=full, max_tokens=self.think_budget,
                                                 temperature=self.temperature, extra_body={**sampling, "chat_template_kwargs": tk})
        m1 = r1.choices[0].message
        think = getattr(m1, "reasoning_content", None) or getattr(m1, "reasoning", None) or ""
        content = m1.content or ""
        finished = r1.choices[0].finish_reason == "stop" and bool(content.strip())
        toks = int(r1.usage.completion_tokens or 0) if r1.usage else 0
        if not finished:                               # thinking hit the budget (or no answer yet): close it and answer
            prefix = "<think>\n" + (think or content) + self.CLOSE_THINK + "\n</think>\n\n"
            kwargs = {"max_tokens": max_tokens, "temperature": self.temperature,
                      "extra_body": {**sampling, "continue_final_message": True, "add_generation_prompt": False, "chat_template_kwargs": tk}}
            if schema is not None:
                kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "action", "schema": schema, "strict": True}}
            r2 = self.client.chat.completions.create(model=self.model, messages=full + [{"role": "assistant", "content": prefix}], **kwargs)
            m2 = r2.choices[0].message
            # the server's reasoning parser never sees a </think> in what is generated here (it is in the prompt),
            # so it files the answer under `reasoning`; take whichever field holds it
            content = m2.content or getattr(m2, "reasoning", None) or getattr(m2, "reasoning_content", None) or ""
            toks += int(r2.usage.completion_tokens or 0) if r2.usage else 0
            think = (think or "") + self.CLOSE_THINK
        self.usage["calls"] += 1; self.usage["seconds"] += time.time() - t0
        if r1.usage:
            self.usage["input_tokens"] += r1.usage.prompt_tokens or 0
        self.usage["output_tokens"] += toks
        self.last_reasoning, self.last_tokens, self.last_seconds = think, toks, round(time.time() - t0, 2)
        return content


class ScriptedProvider:
    model = "scripted"
    CYCLE = [(0, "move_to"), (0, "move_to"), (6, "fire"), (8, "pick_up"), (2, "move_to"), (6, "fire"), (3, "move_to"), (1, "move_to"), (6, "fire"), (7, "wait")]

    def __init__(self):
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}

    def complete(self, system, messages, max_tokens=400, schema=None):
        self.usage["calls"] += 1
        if schema is not None and "lessons" in schema.get("properties", {}):    # the trial-boundary rewrite
            return json.dumps({"lessons": [f"scripted lesson {self.usage['calls']}"], "todo": ["scripted plan"]})
        a, g = self.CYCLE[self.usage["calls"] % len(self.CYCLE)]
        m = re.search(r"You are (?:agent \w, )?at \((\d+), (\d+)\)", messages[-1]["content"]); r, c = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        out = {"why_this_move": "scripted", "mode": "clean", "goal": {"type": g, "cell": [max(r - 2, 0), c], "why": "scripted"}, "action": a}
        if schema is not None and "note" in schema.get("properties", {}):
            out["note"] = ({"op": "add", "id": 0, "text": f"scripted note at call {self.usage['calls']}"} if a == 6 else
                           {"op": "replace", "id": 1, "text": f"scripted revision at call {self.usage['calls']}"} if a == 8 else
                           {"op": "none", "id": 0, "text": ""})
        return json.dumps(out)


# ---------------------------------------------------------------- the episode loop
def run_episode(env, agents, key, args, log, episode=0, recorder=None, on_step=None, init=None, max_steps=None, result=None):
    """on_step(trial, t, state, agents, obs, ks): optional hook called right before the agents decide at each step
    (used by the replay/snapshot tooling; it must not mutate anything).
    init: {"state": env State, "ks": step key} resumes mid-episode from a snapshot (the agents must carry the matching
    memory); max_steps stops after that many env steps; result (a dict) receives the final state and step key."""
    n, inner, outer = env.num_agents, env.num_inner_steps, env.num_outer_steps
    if init is None:
        key, kr, ks = jax.random.split(key, 3)
        _, state = env.reset(kr)
        state = curriculum_start(env, state, args.curriculum)
        if args.reveal_orders:
            given = true_recipes(env, state)
            for ag in agents:
                ag.evidence.reveal(given)
            print("[reveal] told every agent the true recipes: " + "; ".join(f"{Evidence.name(c)} = {list(w)}" for c, w in given.items()), flush=True)
    else:
        state = jax.tree_util.tree_map(jnp.asarray, init["state"]); ks = jnp.asarray(init["ks"])
    step = jax.jit(env.step)
    own_scale = 1.0 / n                                   # the environment pays an apple as n to the agent that ate it
    cleared_in = [[0] * outer for _ in range(n)]; apples_in = [[0.0] * outer for _ in range(n)]
    decisions = [[0] * outer for _ in range(n)]
    if init is not None:                                  # this trial's counters live in the agents' memory
        tr0 = int(state.outer_t)
        for i in range(n):
            cleared_in[i][tr0] = agents[i].tstats["cleared"]; apples_in[i][tr0] = agents[i].tstats["apples"]
    bumps = [0] * n
    cur_trial, done = -1, False
    n_steps = 0
    obs = [observe(env, state, i) for i in range(n)]
    while not done:
        trial, t = int(state.outer_t), int(state.inner_t)
        if trial != cur_trial:
            if cur_trial >= 0:
                # every agent rewrites its notebook: one extra call each, against ~inner*n decisions in the trial
                with ThreadPoolExecutor(max_workers=n) as ex:
                    recs = list(ex.map(lambda i: agents[i].consolidate(cur_trial, outer), range(n)))
                for i, rec in enumerate(recs):
                    log.write(json.dumps({"episode": episode, "agent": i, "trial": cur_trial,
                                          "kind": "notebook", "notebook": rec}) + "\n")
                log.flush()
                for ag in agents:
                    ag.new_trial()
            cur_trial = trial
            obs = [observe(env, state, i) for i in range(n)]

        if on_step is not None:
            on_step(trial, t, state, agents, obs, ks)

        def decide(i):
            return agents[i].decide(obs[i][0], obs[i][1], trial, outer, t, inner, cleared_in[i][trial], apples_in[i][trial])
        with ThreadPoolExecutor(max_workers=n) as ex:
            chosen = list(ex.map(decide, range(n)))
        raw_acts = [ch[0]["action"] for ch in chosen]
        facts_before = [obs[i][1] for i in range(n)]
        grid_before = onp.asarray(state.grid).copy(); locs_before = onp.asarray(state.agent_locs).copy()
        beams = [beam_cells(env, locs_before[i]) if raw_acts[i] == FIRE else [] for i in range(n)]
        ks, k1 = jax.random.split(ks)
        _, state, rew, dones, info = step(k1, state, [RAW_TO_ENV[a] for a in raw_acts])
        n_steps += 1
        done = bool(dones["__all__"]) or (max_steps is not None and n_steps >= max_steps)
        grid_after = onp.asarray(state.grid); locs_after = onp.asarray(state.agent_locs)
        own = onp.asarray(info.get("original_rewards", rew)).reshape(-1) * own_scale
        trial_ended = int(state.inner_t) == 0 and int(state.outer_t) != trial
        obs = [observe(env, state, i) for i in range(n)]
        if recorder is not None:
            recorder.set_caption(f"A0 {chosen[0][0].get('mode')} {chosen[0][0]['action']} - {chosen[0][0]['reasoning'][:70]}")
            recorder(state, {"trial": trial, "t": t, "held": int(onp.asarray(state.held_tool)[0]),
                             "crafted": int(onp.asarray(state.crafted)[0]), "reward": float(own.sum()),
                             "fired_agents": [i for i in range(n) if raw_acts[i] == FIRE]}, trial_ended)
        # What every shot did, cell by cell, as the environment ruled it (report_beam). Each agent's verdicts are
        # its own, so two agents firing on one cell each get an honest record, and no bystander can erase one.
        verdicts = beam_verdicts(info, n)
        shot_effects = {j: verdicts[j] for j in range(n) if raw_acts[j] == FIRE}
        for i, (reply, prompt, raw) in enumerate(chosen):
            fb, fa = facts_before[i], obs[i][1]
            mine = shot_effects.get(i, {})
            my = set(mine); cleared_by_colour, transformed, targets = {}, {}, []
            for (rr, cc), (kind, col, new) in sorted(mine.items()):
                targets.append(col)
                if kind == "cleared":
                    cleared_by_colour[col] = cleared_by_colour.get(col, 0) + 1
                elif kind == "turned":
                    transformed.setdefault(col, {})[new] = transformed.get(col, {}).get(new, 0) + 1
            # river cells in MY view (before the step) that went waste -> water, not by my beam
            cleared_by_others = sum(1 for (rr, cc) in fb["view"] if (rr, cc) not in my
                                    and DIRT_BASE <= int(grid_before[rr, cc]) < DIRT_BASE + NUM_DIRT_TYPES
                                    and int(grid_after[rr, cc]) in (int(Items.river), int(Items.potential_dirt)))
            # shots by agents this agent could see, with their effect on the waste cells it could see
            fired_in_view = {j: [e for rc, e in sorted(eff.items()) if rc in fb["view"]]
                             for j, eff in shot_effects.items() if j != i and (j in fb["others"] or j in fa["others"])}
            moved = raw_acts[i] in (0, 1, 2, 3)
            blocked = moved and tuple(locs_before[i][:2]) == tuple(locs_after[i][:2])
            bumps[i] = bumps[i] + 1 if blocked else 0
            picked = raw_acts[i] == PICK and fb["under"] >= 0 and fa["picks"] != fb["picks"]
            o = {"action": raw_acts[i], "invalid": reply.get("invalid", False), "unavailable": reply.get("unavailable"), "shots": int(raw_acts[i] == FIRE),
                 "cleared": sum(cleared_by_colour.values()), "cleared_by_colour": cleared_by_colour, "transformed": transformed,
                 "shot_targets": targets, "moved_from": [int(x) for x in locs_before[i][:2]], "moved_to": [int(x) for x in locs_after[i][:2]],
                 "facing_after": FACING[int(locs_after[i][2])], "bumps": int(blocked), "bumps3": bumps[i] >= 3,
                 "picked": bool(picked), "picked_tool": fb["under"], "tool_text": fa["tool_text"],
                 "window": onp.asarray(state.pickups)[i].tolist(), "window_clean": fa["picks"],
                 "result": "trial ended" if trial_ended else "done", "steps": 1, "skill": "act", "actions": [raw_acts[i]]}
            agents[i].after_step(t, o, fb, fa, cleared_by_others, fired_in_view, trial_ended, apples=float(own[i]))
            cleared_in[i][trial] += o["cleared"]; apples_in[i][trial] += float(own[i]); decisions[i][trial] += 1
            log.write(json.dumps({"episode": episode, "agent": i, "trial": trial, "t": t, "action": reply, "outcome": o,
                                  "outcome_text": agents[i].last_text, "raw": raw, "prompt": prompt}) + "\n")
        log.flush()
    if recorder is not None:
        recorder.flush(cur_trial)
    per_agent = [[{"return": apples_in[i][tr], "cleared": cleared_in[i][tr]} for tr in range(outer)] for i in range(n)]
    team = [sum(apples_in[i][tr] for i in range(n)) for tr in range(outer)]
    if result is not None:
        result["state"], result["ks"], result["steps"] = state, ks, n_steps
    return per_agent, team, decisions


# ---------------------------------------------------------------- command line
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["vllm", "openai", "scripted"], default="vllm")
    ap.add_argument("--model", default="m"); ap.add_argument("--models", default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:8550/v1"); ap.add_argument("--base-urls", default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--reasoning-effort", default=None, help="openai reasoning models: effort per call (thinking counts against --max-tokens)")
    ap.add_argument("--no-think", action="store_true", help="send chat_template_kwargs.enable_thinking=false (Qwen3.5 thinks by default)")
    ap.add_argument("--think-budget", type=int, default=None, help="thinking ON with a token budget: think at most N tokens, then the think block is closed and the JSON answer is forced (for models without reasoning_effort, e.g. Qwen3.5-4B)")
    ap.add_argument("--think-effort", default=None, choices=["low", "medium", "xhigh"],
                    help="thinking ON with this reasoning_effort (Qwen3.8 template); serve vLLM with --reasoning-parser qwen3")
    ap.add_argument("--history", type=int, default=5, help="recent steps shown")
    ap.add_argument("--lenient", action="store_true", help="retry once on an invalid reply (evaluation); default strict = stay")
    ap.add_argument("--needed-hint", action="store_true", help="ablation: RULES YOU KNOW names the tools this view needs and what your hand is; "
                    "ACTIONS NOW says what the beam would do (rule application done by the harness)")
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--inner", type=int, default=200); ap.add_argument("--outer", type=int, default=4)
    ap.add_argument("--spawn", type=float, default=0.1); ap.add_argument("--skew", type=float, default=0.8)
    ap.add_argument("--apple-threshold", type=float, default=None, help="river dirt fraction below which apples grow (env default 0.4); 1.0 = graded from the first cleared cell")
    ap.add_argument("--on-train", action="store_true")
    ap.add_argument("--tools", choices=["shared", "per-colour"], default="shared",
                    help="shared: 7 working tools (one per hue for light cells + one mid + one dark); per-colour: 15, one per waste colour")
    ap.add_argument("--reveal-orders", action="store_true")
    ap.add_argument("--curriculum", default="none", choices=["none", "river_tool", "station_tool", "half_craft", "first_station", "near_first"])
    ap.add_argument("--seed", type=int, default=906); ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--tag", default="raw2"); ap.add_argument("--log-dir", default="logs/llm"); ap.add_argument("--out-dir", default="logs/llm")
    ap.add_argument("--random-layout", action="store_true", help="a random map per meta-episode from the held-out (or --on-train) layout pool")
    ap.add_argument("--gif-dir", default=None); ap.add_argument("--gif-trials", default="0")
    ap.add_argument("--gif-every", type=int, default=1); ap.add_argument("--gif-fps", type=int, default=8)
    return ap


def make_agents(a, providers):
    rl = bool(getattr(a, "random_layout", False))
    return [Agent(providers[i], P.build_system(a.agents, a.inner, a.outer, me=i, tools=getattr(a, "tools", "shared"),
                                               layout="random" if rl else "fixed"),
                  i, a.agents, history=a.history, max_tokens=a.max_tokens, strict=not a.lenient, hint=a.needed_hint,
                  random_layout=rl)
            for i in range(a.agents)]


def main():
    a = build_parser().parse_args()
    env = make_env(a.agents, a.inner, a.outer, a.spawn, a.skew, a.on_train, apple_threshold=a.apple_threshold, tools=a.tools,
                   layouts="random" if a.random_layout else "fixed")
    if a.provider == "scripted":
        providers = [ScriptedProvider() for _ in range(a.agents)]
    else:
        models = (a.models or a.model).split(","); urls = (a.base_urls or a.base_url).split(",")
        api_key = "EMPTY"
        if a.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY") or open(os.path.expanduser("~/.config/openai/key")).read().strip()
            urls = ["https://api.openai.com/v1"]
        extra = ({"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": a.think_effort}} if a.think_effort else
                 {"chat_template_kwargs": {"enable_thinking": False}} if a.no_think else None)
        providers = [ChatProvider(models[i % len(models)].strip(), urls[i % len(urls)].strip(), api_key, a.temperature,
                                  extra_body=extra, reasoning_effort=a.reasoning_effort, think_budget=a.think_budget)
                     for i in range(a.agents)]
    os.makedirs(a.log_dir, exist_ok=True); os.makedirs(a.out_dir, exist_ok=True)
    log_path = os.path.join(a.log_dir, f"{a.tag}_decisions.jsonl")
    print(f"{providers[0].model} | cleanup prompt | {a.episodes} episodes of {a.outer}x{a.inner} | log {log_path}", flush=True)
    results = []
    with open(log_path, "w") as log:
        for ep in range(a.episodes):
            agents = make_agents(a, providers)
            if ep == 0:
                open(os.path.join(a.log_dir, f"{a.tag}_system.txt"), "w").write(agents[0].system)
            recorder = None
            if a.gif_dir and ep == 0:
                from llm_policy.record import GifRecorder
                recorder = GifRecorder(env, a.gif_dir, a.tag, trials=[int(x) for x in a.gif_trials.split(",")], every=a.gif_every, fps=a.gif_fps)
            t0 = time.time()
            try:
                per_agent, team, decisions = run_episode(env, agents, jax.random.PRNGKey(a.seed + ep), a, log, ep, recorder)
            except BaseException:
                if recorder is not None and recorder.frames:      # keep the partial trial (e.g. API quota hit)
                    recorder.flush(recorder.current or 0)
                raise
            usage = {k: sum(p.usage[k] for p in providers) for k in providers[0].usage}
            plan = [{"goals": ag.plan.n_goals, "consistent": ag.plan.consistent, "inconsistent": ag.plan.inconsistent,
                     "changes": ag.plan.changes, "changes_without_event": ag.plan.changes_without_event, "modes": ag.mode_counts} for ag in agents]
            nb = [{"notes": ag.notebook.n_notes, "rewrites_ok": sum(r["ok"] for r in ag.notebook.history),
                   "rewrites_failed": sum(not r["ok"] for r in ag.notebook.history),
                   "lessons_now": list(ag.notebook.lessons), "todo_now": list(ag.notebook.todo)}
                  for ag in agents]
            results.append({"seed": a.seed + ep, "team": team, "per_agent": per_agent, "decisions": decisions, "notebook": nb,
                            "parse_failures": [ag.parse_failures for ag in agents], "unavailable_pickups": [ag.unavailable for ag in agents], "plan": plan, "usage": usage, "seconds": time.time() - t0})
            print(f"episode {ep}: TEAM apples/trial {[round(x) for x in team]} | cleared/trial "
                  f"{[sum(per_agent[i][tr]['cleared'] for i in range(a.agents)) for tr in range(a.outer)]} | invalid replies {[ag.parse_failures for ag in agents]} | pick-up off-station {[ag.unavailable for ag in agents]} | "
                  f"goal-consistent {sum(p['consistent'] for p in plan)}/{sum(p['consistent'] + p['inconsistent'] for p in plan)} | "
                  f"goal changes w/o event {sum(p['changes_without_event'] for p in plan)} | modes {[p['modes'] for p in plan]} | "
                  + f"notes {[x['notes'] for x in nb]}, rewrites ok {sum(x['rewrites_ok'] for x in nb)}/"
                    f"{sum(x['rewrites_ok'] + x['rewrites_failed'] for x in nb)} | "
                  + f"{usage['calls']} calls, {time.time() - t0:.0f}s", flush=True)
            json.dump({"config": vars(a), "episodes": results}, open(os.path.join(a.out_dir, f"llm_{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
