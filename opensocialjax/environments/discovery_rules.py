"""Rule encodings for the ``*_discovery`` environments, XLand style.

XLand-MiniGrid's core move is that a task is *data, not code*: every rule is
a fixed-length integer row ``[rule_id, arg, arg, ...]``, a ruleset is a
``(MAX_RULES, RULE_ENCODING_LEN)`` array of such rows, and the step function
*interprets* the rows at runtime (``lax.scan`` over rows, ``lax.switch`` on
``rule_id``). Three consequences, and they are the whole point:

- the ruleset lives in ``State`` as a plain array, so ``vmap`` over
  environments gives every parallel environment its own task for free;
- rulesets of different logical sizes share one physical shape, because rows
  are padded with the id-0 ``EmptyRule`` sentinel — a no-op branch in the
  switch, so padding costs nothing under jit;
- new rule *types* are new rows in the switch table, not new code paths in
  the environment.

This module is the discovery-environment analogue of xland-minigrid's
``core/rules.py``. The vocabulary indexes it speaks (waste-type index, tool
index) are the environment's business; nothing here imports an environment.

Encoding
--------

Each row is ``RULE_ENCODING_LEN`` uint8 values ``[rule_id, *args]``:

====  ===================  ===================================================
 id   name                 semantics of ``args``
====  ===================  ===================================================
 0    EmptyRule            none — padding sentinel, never fires
 1    ToolClearsRule       ``(waste, tool)`` — the beam, fired at waste colour
                          ``waste`` while holding ``tool``, clears it to water
 2    ToolTransformsRule   ``(waste, tool, next)`` — the same beam turns the
                          waste into colour ``next`` instead: XLand's
                          production rule ``(a, tool) -> b``, the link of a
                          cleaning CHAIN whose last link is a ToolClearsRule
 3    CraftSequenceRule    ``(t0, t1, t2)`` — the ORDER of station pickups
                          that hands the agent the cleaning tool; the beam
                          then clears waste of every colour. Interpreted by
                          the environment's pickup logic, not by the beam.
 4    HueClearsRule        ``(hue, tool, num_shades)`` — colours read as
                          ``hue * num_shades + shade``; the beam, holding
                          ``tool``, clears the LIGHT (shade 0) waste of
                          ``hue`` to water
 5    ShadeLightensRule    ``(shade, tool, num_shades)`` — the beam, holding
                          ``tool``, turns waste of that shade (any hue) one
                          shade lighter: colour ``c`` becomes ``c - 1``
 6    CraftRecipeRule      ``(product, t0, t1)`` — entering stations ``t0``
                          then ``t1`` (sliding window) puts crafted tool
                          ``product`` in the agent's hand; the Hue/Shade rows
                          above then name crafted ids as their ``tool``.
                          Interpreted by the pickup logic, not by the beam.
====  ===================  ===================================================

A vanilla ``open_cleanup`` task (every waste type clearable by exactly
one tool) is six ``ToolClearsRule`` rows plus ``EmptyRule`` padding. The
encoding is deliberately more general than that: a ruleset with a waste type
that *no* row clears, or that two tools clear, is representable without
touching this module — which is what "extending the task space = widening the
data, not adding code" means in practice.
"""

from __future__ import annotations

import abc

import jax
import jax.numpy as jnp
from flax import struct

# [rule_id, arg0, arg1, arg2] — one spare arg so a future rule type with three
# arguments does not force a benchmark-breaking reshape.
RULE_ENCODING_LEN = 4
# Rows per ruleset. Vanilla open_cleanup uses NUM_DIRT_COLORS = 15 of them;
# the slack is EmptyRule padding today and headroom for distractor or
# production rules tomorrow.
MAX_RULES = 16

# rule ids — index into the lax.switch table in check_clean, so order matters
# and EmptyRule MUST stay at 0 (zero-padding == EmptyRule is the invariant the
# whole padding scheme rests on).
EMPTY = 0
TOOL_CLEARS = 1
TOOL_TRANSFORMS = 2
CRAFT_SEQUENCE = 3
HUE_CLEARS = 4
SHADE_LIGHTENS = 5
CRAFT_RECIPE = 6
# open_harvest rows — resource dynamics, not the beam's business; the
# beam tables below give them no-op branches.
REGROW_RATE = 7      # [7, hue, order_idx, 0]: PAY_ORDERS[order_idx][s] = speed rank of a cell whose last apple left at stage s; x REGROW_RATES[rank]
RIPE_PEAK = 8        # [8, hue, order_idx, 0]: PAY_ORDERS[order_idx] = pay rank of each ripeness stage (rank 0 pays STAGE_PAY[0] = 1, ...)
# Every lax.switch table in this module must have exactly this many branches.
# lax.switch CLAMPS an out-of-range index, so a rule id without a branch would
# silently take the last (no-op) branch instead of raising.
# [9, colour, t0, t1]: the pick-up order (t0 then t1) crafts the working tool for ONE waste colour, whose id is
# TYPE_TOOL_BASE + colour; that tool clears the colour if it is light (shade 0) and otherwise turns it one shade
# lighter, and does nothing to any other colour (the per-colour variant: 15 recipes instead of 7)
TYPE_CRAFT = 9
TYPE_TOOL_BASE = 32
TYPE_NUM_SHADES = 3
NUM_RULE_IDS = 10
# apply_beam returns this when no rule fires (the cell is left as it is)
NO_EFFECT = -1


class BaseRule(struct.PyTreeNode):
    """A rule answers one question: does the cleaning beam clear this cell?

    ``__call__(waste_type, tool) -> bool`` — whether *this* rule says a beam
    fired while holding ``tool`` clears waste of type ``waste_type``. Rules
    are OR-combined by ``check_clean``, mirroring how xland-minigrid's
    ``check_goal`` interprets goal encodings.
    """

    @abc.abstractmethod
    def __call__(self, waste_type: jax.Array, tool: jax.Array) -> jax.Array: ...

    @classmethod
    @abc.abstractmethod
    def decode(cls, encoding: jax.Array) -> "BaseRule": ...

    @abc.abstractmethod
    def encode(self) -> jax.Array: ...


class EmptyRule(BaseRule):
    """The padding sentinel: never fires, encodes to all zeros."""

    def __call__(self, waste_type, tool):
        return jnp.asarray(False)

    @classmethod
    def decode(cls, encoding):
        return cls()

    def encode(self):
        return jnp.zeros(RULE_ENCODING_LEN, dtype=jnp.uint8)


class ToolClearsRule(BaseRule):
    """``tool`` clears waste of type ``waste_type``; any other pairing is not
    this rule's business (another row may still allow it)."""

    waste_type: jax.Array
    tool: jax.Array

    def __call__(self, waste_type, tool):
        return jnp.asarray(
            (self.waste_type == waste_type) & (self.tool == tool)
        )

    @classmethod
    def decode(cls, encoding):
        return cls(waste_type=encoding[1], tool=encoding[2])

    def encode(self):
        return (
            jnp.zeros(RULE_ENCODING_LEN, dtype=jnp.uint8)
            .at[0].set(TOOL_CLEARS)
            .at[1].set(jnp.asarray(self.waste_type, dtype=jnp.uint8))
            .at[2].set(jnp.asarray(self.tool, dtype=jnp.uint8))
        )


class ToolTransformsRule(BaseRule):
    """``tool`` turns waste of colour ``waste_type`` into colour ``next``.

    The production-rule link of a cleaning chain (XLand's ``a -> b``): the
    beam does not clear the cell, it changes what is there, and a later
    ToolClearsRule on ``next`` finishes the job. ``__call__`` reports only
    whether this rule *matches*; ``apply_beam`` turns a match into the new
    cell code.
    """

    waste_type: jax.Array
    tool: jax.Array
    next: jax.Array

    def __call__(self, waste_type, tool):
        return jnp.asarray(
            (self.waste_type == waste_type) & (self.tool == tool)
        )

    @classmethod
    def decode(cls, encoding):
        return cls(waste_type=encoding[1], tool=encoding[2], next=encoding[3])

    def encode(self):
        return (
            jnp.zeros(RULE_ENCODING_LEN, dtype=jnp.uint8)
            .at[0].set(TOOL_TRANSFORMS)
            .at[1].set(jnp.asarray(self.waste_type, dtype=jnp.uint8))
            .at[2].set(jnp.asarray(self.tool, dtype=jnp.uint8))
            .at[3].set(jnp.asarray(self.next, dtype=jnp.uint8))
        )


class CraftSequenceRule(BaseRule):
    """The hidden rule of the crafting variant: pick tools up in this ORDER
    (the last ``len(sequence)`` pickups, as a sliding window) and you hold the
    cleaning tool. It never fires through the beam interpreter — crafting is
    checked where pickups happen — so ``__call__`` is always False."""

    sequence: jax.Array

    def __call__(self, waste_type, tool):
        return jnp.asarray(False)

    @classmethod
    def decode(cls, encoding, k=RULE_ENCODING_LEN - 1):
        return cls(sequence=encoding[1:1 + k])

    def encode(self):
        return encode_craft_sequence(self.sequence)[0]


def _hue_shade(waste_type, num_shades):
    """Split a colour index into ``(hue, shade)``: colour = hue * num_shades
    + shade. ``num_shades`` comes from the rule row, so the interpreter needs
    no constant from the environment."""
    s = jnp.maximum(jnp.asarray(num_shades, dtype=jnp.int32), 1)
    w = jnp.asarray(waste_type, dtype=jnp.int32)
    return w // s, w % s


class HueClearsRule(BaseRule):
    """``tool`` clears the LIGHT (shade 0) waste of hue ``hue`` to water.

    One half of the attribute-chain rule (the other is ShadeLightensRule):
    the table is indexed by hue, not by colour, so five rows cover fifteen
    colours. Row ``[4, hue, tool, num_shades]``.
    """

    hue: jax.Array
    tool: jax.Array
    num_shades: jax.Array

    def __call__(self, waste_type, tool):
        h, s = _hue_shade(waste_type, self.num_shades)
        return jnp.asarray((h == self.hue) & (s == 0) & (self.tool == tool))

    @classmethod
    def decode(cls, encoding):
        return cls(hue=encoding[1], tool=encoding[2], num_shades=encoding[3])

    def encode(self):
        return (
            jnp.zeros(RULE_ENCODING_LEN, dtype=jnp.uint8)
            .at[0].set(HUE_CLEARS)
            .at[1].set(jnp.asarray(self.hue, dtype=jnp.uint8))
            .at[2].set(jnp.asarray(self.tool, dtype=jnp.uint8))
            .at[3].set(jnp.asarray(self.num_shades, dtype=jnp.uint8))
        )


class ShadeLightensRule(BaseRule):
    """``tool`` turns waste of shade ``shade`` — whatever its hue — one shade
    lighter: colour ``c`` becomes ``c - 1``. Shared across hues, which is
    what makes evidence on one hue carry to the others. Row
    ``[5, shade, tool, num_shades]`` with ``shade >= 1``. Like
    ToolTransformsRule, ``__call__`` reports a match; ``apply_beam`` turns it
    into the new cell code.
    """

    shade: jax.Array
    tool: jax.Array
    num_shades: jax.Array

    def __call__(self, waste_type, tool):
        _, s = _hue_shade(waste_type, self.num_shades)
        return jnp.asarray((s == self.shade) & (s > 0) & (self.tool == tool))

    @classmethod
    def decode(cls, encoding):
        return cls(shade=encoding[1], tool=encoding[2], num_shades=encoding[3])

    def encode(self):
        return (
            jnp.zeros(RULE_ENCODING_LEN, dtype=jnp.uint8)
            .at[0].set(SHADE_LIGHTENS)
            .at[1].set(jnp.asarray(self.shade, dtype=jnp.uint8))
            .at[2].set(jnp.asarray(self.tool, dtype=jnp.uint8))
            .at[3].set(jnp.asarray(self.num_shades, dtype=jnp.uint8))
        )


class CraftRecipeRule(BaseRule):
    """Attribute-craft variant: entering stations in the order ``sequence``
    (the last ``len(sequence)`` pickups, a sliding window) puts crafted tool
    ``product`` in the agent's one hand — the hand that otherwise holds the
    base tool of the last station. Row ``[6, product, t0, t1]``. Never fires
    through the beam interpreter."""

    product: jax.Array
    sequence: jax.Array

    def __call__(self, waste_type, tool):
        return jnp.asarray(False)

    @classmethod
    def decode(cls, encoding, k=RULE_ENCODING_LEN - 2):
        return cls(product=encoding[1], sequence=encoding[2:2 + k])

    def encode(self):
        row = jnp.zeros(RULE_ENCODING_LEN, dtype=jnp.uint8).at[0].set(CRAFT_RECIPE)
        row = row.at[1].set(jnp.asarray(self.product, dtype=jnp.uint8))
        seq = jnp.asarray(self.sequence, dtype=jnp.uint8)
        return row.at[2:2 + seq.shape[0]].set(seq)


def encode_attr_craft_ruleset(recipes, num_tools: int, num_hues: int = 5, num_shades: int = 3,
                              max_rules: int = MAX_RULES) -> jax.Array:
    """The attribute-craft rule: every acting tool is CRAFTED. Crafted tool
    ids follow the base tools — ``num_tools + h`` clears light waste of hue
    ``h``, ``num_tools + num_hues + (s - 1)`` lightens shade ``s`` — and
    ``recipes[j]`` (``(num_hues + num_shades - 1, k)`` station orders) is
    the pickup order that crafts tool ``num_tools + j``. Rows: the Hue /
    Shade tables naming the crafted ids, then one CraftRecipeRule per
    product. jit/vmap-friendly."""
    recipes = jnp.asarray(recipes, dtype=jnp.uint8)
    n, k = recipes.shape
    assert n == num_hues + num_shades - 1 and k <= RULE_ENCODING_LEN - 2
    prod = (num_tools + jnp.arange(n)).astype(jnp.uint8)
    attr_rows = encode_attr_tables(prod[:num_hues], prod[num_hues:], num_shades, max_rules=n)
    recipe_rows = (jnp.zeros((n, RULE_ENCODING_LEN), dtype=jnp.uint8)
                   .at[:, 0].set(CRAFT_RECIPE).at[:, 1].set(prod).at[:, 2:2 + k].set(recipes))
    return pad_ruleset(jnp.concatenate([attr_rows, recipe_rows], axis=0), max_rules)


def decode_craft_recipes(encodings: jax.Array, k: int = 2):
    """``(valid (MAX_RULES,), products (MAX_RULES,), sequences (MAX_RULES, k))``
    of the CraftRecipeRule rows; ``-1`` outside valid rows."""
    typed = encodings[:, 0] == TYPE_CRAFT
    valid = (encodings[:, 0] == CRAFT_RECIPE) | typed
    products = jnp.where(typed, TYPE_TOOL_BASE + encodings[:, 1].astype(jnp.int16),
                         jnp.where(valid, encodings[:, 1], -1)).astype(jnp.int16)
    seqs = jnp.where(valid[:, None], encodings[:, 2:2 + k], -1).astype(jnp.int16)
    return valid, products, seqs


def encode_type_craft_ruleset(recipes, max_rules: int = MAX_RULES) -> jax.Array:
    """The per-colour craft rule: ``recipes[c]`` (``(num_colours, k)`` station orders) is the pick-up order that
    crafts the working tool for waste colour ``c`` (tool id ``TYPE_TOOL_BASE + c``). One row per colour, the effect
    is implied by the colour's shade. jit/vmap-friendly."""
    recipes = jnp.asarray(recipes, dtype=jnp.uint8)
    n, k = recipes.shape
    assert n <= max_rules and k <= RULE_ENCODING_LEN - 2
    rows = (jnp.zeros((n, RULE_ENCODING_LEN), dtype=jnp.uint8)
            .at[:, 0].set(TYPE_CRAFT).at[:, 1].set(jnp.arange(n, dtype=jnp.uint8)).at[:, 2:2 + k].set(recipes))
    return pad_ruleset(rows, max_rules)


def sample_type_craft_ruleset(key, num_tools: int, k: int = 2, num_colours: int = 15) -> jax.Array:
    """Uniform over per-colour craft rulesets: ``num_colours`` distinct orders, a prefix of a random permutation of
    all ``num_tools**k`` orders."""
    assert num_tools ** k >= num_colours, "not enough pick-up orders for one distinct recipe per colour"
    ids = jax.random.permutation(key, num_tools ** k)[:num_colours]
    digits = jnp.stack([(ids // (num_tools ** (k - 1 - i))) % num_tools for i in range(k)], axis=-1)
    return encode_type_craft_ruleset(digits)


def num_type_craft_rulesets(num_tools: int, k: int = 2, num_colours: int = 15) -> int:
    """P(num_tools**k, num_colours): 20,922,789,888,000 at four tools, two pick-ups, 15 colours."""
    m, total = num_tools ** k, 1
    for i in range(num_colours):
        total *= m - i
    return total


def num_attr_craft_rulesets(num_tools: int, k: int = 2, num_hues: int = 5, num_shades: int = 3) -> int:
    """Distinct recipes for the ``num_hues + num_shades - 1`` products:
    P(num_tools**k, n) — 57,657,600 at four tools, two pickups."""
    m, n, total = num_tools ** k, num_hues + num_shades - 1, 1
    for i in range(n):
        total *= m - i
    return total


def sample_attr_craft_ruleset(key, num_tools: int, k: int = 2, num_hues: int = 5,
                              num_shades: int = 3) -> jax.Array:
    """Uniform over attribute-craft rulesets, pure jax: the products'
    recipes are distinct orders (a prefix of a random permutation of all
    ``num_tools**k`` orders), decoded to base-``num_tools`` digits."""
    n = num_hues + num_shades - 1
    ids = jax.random.permutation(key, num_tools ** k)[:n]
    digits = jnp.stack([(ids // (num_tools ** (k - 1 - i))) % num_tools for i in range(k)], axis=-1)
    return encode_attr_craft_ruleset(digits, num_tools, num_hues, num_shades)


def encode_attr_tables(hue_tools, shade_tools, num_shades: int = 3,
                       max_rules: int = MAX_RULES) -> jax.Array:
    """Encode the attribute-chain rule: ``hue_tools[h]`` clears light waste
    of hue ``h``; ``shade_tools[i]`` lightens shade ``i + 1`` (so it has
    ``num_shades - 1`` entries: mid, dark, ...). jit/vmap-friendly."""
    hue_tools = jnp.asarray(hue_tools, dtype=jnp.uint8)
    shade_tools = jnp.asarray(shade_tools, dtype=jnp.uint8)
    nh, ns = hue_tools.shape[0], shade_tools.shape[0]
    hue_rows = jnp.stack([
        jnp.full((nh,), HUE_CLEARS, dtype=jnp.uint8),
        jnp.arange(nh, dtype=jnp.uint8),
        hue_tools,
        jnp.full((nh,), num_shades, dtype=jnp.uint8),
    ], axis=-1)
    shade_rows = jnp.stack([
        jnp.full((ns,), SHADE_LIGHTENS, dtype=jnp.uint8),
        jnp.arange(1, ns + 1, dtype=jnp.uint8),
        shade_tools,
        jnp.full((ns,), num_shades, dtype=jnp.uint8),
    ], axis=-1)
    return pad_ruleset(jnp.concatenate([hue_rows, shade_rows], axis=0), max_rules)


def decode_attr_tables(encodings: jax.Array, num_hues: int = 5, num_shades: int = 3):
    """Read the attribute-chain rule back: ``(hue_tools (num_hues,),
    shade_tools (num_shades,))`` as int16, ``-1`` where no row acts —
    ``shade_tools[0]`` is always ``-1`` (light waste is cleared by hue, not
    lightened). Non-attribute rows scatter into a dump slot and are dropped."""
    kind = encodings[:, 0]
    hidx = jnp.where(kind == HUE_CLEARS, encodings[:, 1], num_hues).astype(jnp.int32)
    hue_tools = jnp.full((num_hues + 1,), -1, dtype=jnp.int16)
    hue_tools = hue_tools.at[hidx].set(encodings[:, 2].astype(jnp.int16))[:num_hues]
    sidx = jnp.where(kind == SHADE_LIGHTENS, encodings[:, 1], num_shades).astype(jnp.int32)
    shade_tools = jnp.full((num_shades + 1,), -1, dtype=jnp.int16)
    shade_tools = shade_tools.at[sidx].set(encodings[:, 2].astype(jnp.int16))[:num_shades]
    return hue_tools, shade_tools


def is_attr_ruleset(encodings: jax.Array) -> jax.Array:
    return (encodings[:, 0] == HUE_CLEARS).any()


def decode_attr_chain(encodings: jax.Array, num_waste_types: int, num_shades: int = 3):
    """The attribute rule as per-colour tables in ``decode_chain``'s format:
    ``tools[c]`` = the tool that acts on colour ``c`` (its hue's clearing tool
    for light waste, its shade's lightening tool otherwise); ``nexts[c]`` =
    ``-1`` (water) for light waste, ``c - 1`` for the rest."""
    num_hues = num_waste_types // num_shades
    hue_tools, shade_tools = decode_attr_tables(encodings, num_hues, num_shades)
    c = jnp.arange(num_waste_types)
    h, s = c // num_shades, c % num_shades
    tools = jnp.where(s == 0, hue_tools[jnp.clip(h, 0, num_hues - 1)], shade_tools[s])
    nexts = jnp.where(s == 0, -1, c - 1)
    return tools.astype(jnp.int16), nexts.astype(jnp.int16)


def all_attr_tables(num_tools: int, num_hues: int = 5, num_shades: int = 3):
    """Every attribute rule over ``num_tools`` tools: the ``num_shades - 1``
    shade tools are an ordered choice of DISTINCT tools (no single tool
    lightens everything), the hue tools are unconstrained. Returns
    ``(hue_tools (N, num_hues), shade_tools (N, num_shades - 1))`` with
    ``N = P(num_tools, num_shades - 1) * num_tools ** num_hues``."""
    import itertools
    import numpy as onp
    hs, ss = [], []
    for shade_tools in itertools.permutations(range(num_tools), num_shades - 1):
        for hue_tools in itertools.product(range(num_tools), repeat=num_hues):
            hs.append(hue_tools)
            ss.append(shade_tools)
    return onp.asarray(hs, dtype=onp.uint8), onp.asarray(ss, dtype=onp.uint8)


def all_attr_rulesets(num_tools: int, num_hues: int = 5, num_shades: int = 3) -> jax.Array:
    hs, ss = all_attr_tables(num_tools, num_hues, num_shades)
    return jax.vmap(lambda h, s: encode_attr_tables(h, s, num_shades))(
        jnp.asarray(hs), jnp.asarray(ss))


def sample_attr_ruleset(key, num_tools: int, num_hues: int = 5, num_shades: int = 3) -> jax.Array:
    """Uniform over ``all_attr_rulesets``, in pure jax (vmappable): the shade
    tools are the prefix of a random permutation of the tools (distinct by
    construction), the hue tools are independent uniform draws."""
    k_perm, k_hue = jax.random.split(key)
    perm = jax.random.permutation(k_perm, num_tools)
    shade_tools = perm[:num_shades - 1]
    hue_tools = jax.random.randint(k_hue, (num_hues,), 0, num_tools)
    return encode_attr_tables(hue_tools, shade_tools, num_shades)


def encode_craft_sequence(seq, max_rules: int = MAX_RULES) -> jax.Array:
    """One CraftSequenceRule row ``[3, t0, t1, ...]`` plus EmptyRule padding."""
    seq = jnp.asarray(seq, dtype=jnp.uint8)
    row = jnp.zeros(RULE_ENCODING_LEN, dtype=jnp.uint8).at[0].set(CRAFT_SEQUENCE)
    row = row.at[1:1 + seq.shape[0]].set(seq)
    return pad_ruleset(row[None, :], max_rules)


def decode_craft_sequence(encodings: jax.Array, k: int) -> jax.Array:
    """The pickup order of the first CraftSequenceRule row, or all -1."""
    is_cs = encodings[:, 0] == CRAFT_SEQUENCE
    row = encodings[jnp.argmax(is_cs)]
    return jnp.where(is_cs.any(), row[1:1 + k].astype(jnp.int16),
                     -jnp.ones((k,), dtype=jnp.int16))


def all_craft_sequences(num_tools: int, k: int):
    """Every length-``k`` pickup order over ``num_tools`` tools, repeats
    allowed: ``num_tools ** k`` rows, in lexicographic order."""
    import itertools
    import numpy as onp
    return onp.asarray(list(itertools.product(range(num_tools), repeat=k)), dtype=onp.uint8)


def all_craft_rulesets(num_tools: int, k: int) -> jax.Array:
    return jax.vmap(encode_craft_sequence)(jnp.asarray(all_craft_sequences(num_tools, k)))


def sample_craft_ruleset(key, num_tools: int, k: int) -> jax.Array:
    return encode_craft_sequence(jax.random.randint(key, (k,), 0, num_tools).astype(jnp.uint8))


def check_clean(encodings: jax.Array, waste_type: jax.Array, tool: jax.Array) -> jax.Array:
    """Interpret a ruleset: does any row clear ``waste_type`` with ``tool``?

    ``encodings`` is ``(MAX_RULES, RULE_ENCODING_LEN)``. Like xland-minigrid's
    ``check_rule``/``check_goal``, this scans the rows and dispatches each on
    its ``rule_id`` with ``lax.switch`` — every branch is compiled once, and
    EmptyRule padding rows take the no-op branch. Cost is bounded by
    MAX_RULES regardless of how many rules the task logically has.
    """

    def _check(cleared, encoding):
        branches = (
            lambda: EmptyRule.decode(encoding)(waste_type, tool),
            lambda: ToolClearsRule.decode(encoding)(waste_type, tool),
            # a transform is not a clear
            lambda: jnp.asarray(False),
            # crafting is not the beam's business
            lambda: jnp.asarray(False),
            lambda: HueClearsRule.decode(encoding)(waste_type, tool),
            # a lighten is not a clear
            lambda: jnp.asarray(False),
            # a recipe is the pickup logic's business
            lambda: jnp.asarray(False),
            # harvest rows: regrowth rate / ripeness peak — nothing to clear
            lambda: jnp.asarray(False),
            lambda: jnp.asarray(False),
            # per-colour tool: clears its own colour, and only a light one
            lambda: (jnp.asarray(waste_type) == encoding[1]) & (jnp.asarray(tool) == TYPE_TOOL_BASE + encoding[1].astype(jnp.int32))
                    & (jnp.asarray(waste_type) % TYPE_NUM_SHADES == 0),
        )
        assert len(branches) == NUM_RULE_IDS
        row = jax.lax.switch(encoding[0], branches)
        return cleared | row, None

    cleared, _ = jax.lax.scan(_check, jnp.asarray(False), encodings)
    return cleared


def apply_beam(encodings, waste_type, tool, water_code, dirt_base):
    """Interpret a ruleset as a TRANSITION: what does the cell become when
    the beam hits waste colour ``waste_type`` while holding ``tool``?

    Returns the new grid code — ``water_code`` for a ToolClearsRule match,
    ``dirt_base + next`` for a ToolTransformsRule match — or ``NO_EFFECT``
    when no row fires. The first matching row wins, scanned in order, with
    ``lax.switch`` on ``rule_id`` exactly as ``check_clean`` does; rulesets
    generated here never contain two rows for one colour, so order is moot.
    """

    def _row(carry, encoding):
        matched, result = carry
        branches = (
            lambda: (jnp.asarray(False), jnp.asarray(NO_EFFECT, dtype=jnp.int32)),
            lambda: (ToolClearsRule.decode(encoding)(waste_type, tool),
                     jnp.asarray(water_code, dtype=jnp.int32)),
            lambda: (ToolTransformsRule.decode(encoding)(waste_type, tool),
                     jnp.asarray(dirt_base, dtype=jnp.int32)
                     + encoding[3].astype(jnp.int32)),
            lambda: (jnp.asarray(False), jnp.asarray(NO_EFFECT, dtype=jnp.int32)),
            lambda: (HueClearsRule.decode(encoding)(waste_type, tool),
                     jnp.asarray(water_code, dtype=jnp.int32)),
            # one shade lighter, same hue: colour c -> c - 1 (only taken on a
            # match, and a match needs shade >= 1, so the result stays waste)
            lambda: (ShadeLightensRule.decode(encoding)(waste_type, tool),
                     jnp.asarray(dirt_base, dtype=jnp.int32)
                     + jnp.asarray(waste_type, dtype=jnp.int32) - 1),
            lambda: (jnp.asarray(False), jnp.asarray(NO_EFFECT, dtype=jnp.int32)),
            # harvest rows: no-op for the beam
            lambda: (jnp.asarray(False), jnp.asarray(NO_EFFECT, dtype=jnp.int32)),
            lambda: (jnp.asarray(False), jnp.asarray(NO_EFFECT, dtype=jnp.int32)),
            # per-colour tool: acts on its own colour only -- light -> water, mid/dark -> one shade lighter
            lambda: ((jnp.asarray(waste_type).astype(jnp.int32) == encoding[1].astype(jnp.int32))
                     & (jnp.asarray(tool).astype(jnp.int32) == TYPE_TOOL_BASE + encoding[1].astype(jnp.int32)),
                     jnp.where(jnp.asarray(waste_type).astype(jnp.int32) % TYPE_NUM_SHADES == 0,
                               jnp.asarray(water_code, dtype=jnp.int32),
                               jnp.asarray(dirt_base, dtype=jnp.int32) + jnp.asarray(waste_type, dtype=jnp.int32) - 1)),
        )
        assert len(branches) == NUM_RULE_IDS
        hit, out = jax.lax.switch(encoding[0], branches)
        take = hit & ~matched
        return (matched | hit, jnp.where(take, out, result)), None

    (_, result), _ = jax.lax.scan(
        _row, (jnp.asarray(False), jnp.asarray(NO_EFFECT, dtype=jnp.int32)),
        encodings)
    return result


def pad_ruleset(rows: jax.Array, max_rules: int = MAX_RULES) -> jax.Array:
    """Pad ``(n, RULE_ENCODING_LEN)`` rows to ``(max_rules, ...)`` with zero
    rows — and zero rows *are* EmptyRule, which is why padding is free."""
    return jnp.pad(rows, ((0, max_rules - rows.shape[0]), (0, 0)))


def count_rules(encodings: jax.Array) -> jax.Array:
    """Number of non-Empty rows — the ruleset's logical size, kept alongside
    the padded encoding the way xland benchmarks keep ``num_rules``."""
    return jnp.sum(encodings[..., 0] != EMPTY, axis=-1).astype(jnp.uint8)


def encode_tool_table(table: jax.Array, max_rules: int = MAX_RULES) -> jax.Array:
    """Encode a dense compatibility table (``table[k]`` = the tool clearing
    waste type ``k``) as a padded ruleset: one ToolClearsRule row per type.

    jit/vmap-friendly, so a whole benchmark can be encoded in one vmap.
    """
    k = jnp.arange(table.shape[0], dtype=jnp.uint8)
    rows = jnp.stack(
        [
            jnp.full_like(k, TOOL_CLEARS),
            k,
            table.astype(jnp.uint8),
            jnp.zeros_like(k),
        ],
        axis=-1,
    )
    return pad_ruleset(rows, max_rules)


def decode_tool_table(encodings: jax.Array, num_waste_types: int) -> jax.Array:
    """Read a ruleset back into a dense table: ``table[k]`` = the tool that
    clears waste type ``k``, or ``-1`` if no row clears it.

    The inverse of ``encode_tool_table`` for rulesets made of ToolClearsRule
    rows; diagnostics and scripted oracles read the rule through this. A dense
    table cannot express two tools clearing the same waste type — if rows
    conflict, the later row wins here while ``check_clean`` lets either fire —
    so this is exact for the vanilla one-tool-per-type space and a projection
    beyond it.
    """
    is_tc = encodings[:, 0] == TOOL_CLEARS
    # non-ToolClears rows scatter into a dump slot past the table's end
    idx = jnp.where(is_tc, encodings[:, 1], num_waste_types).astype(jnp.int32)
    table = jnp.full((num_waste_types + 1,), -1, dtype=jnp.int16)
    table = table.at[idx].set(encodings[:, 2].astype(jnp.int16))
    return table[:num_waste_types]


def decode_chain(encodings: jax.Array, num_waste_types: int, num_shades: int = 3):
    """Read a chain ruleset back into two dense tables.

    ``tools[k]`` — the tool the beam must carry to act on waste colour ``k``
    (``-1`` if no row acts on it); ``nexts[k]`` — what the cell becomes:
    ``-1`` for water (a ToolClearsRule) or the colour a ToolTransformsRule
    turns it into. A ruleset of ToolClearsRule rows only gives
    ``nexts == -1`` everywhere and ``tools == decode_tool_table``. An
    attribute ruleset (HueClearsRule / ShadeLightensRule rows) is expanded
    per colour through ``decode_attr_chain``, so every consumer of this
    table — the environment's ``rule_table`` / ``rule_next``, scripted
    oracles, ``chain_depth_of`` — reads it without knowing the variant.
    """
    kind = encodings[:, 0]
    acts = (kind == TOOL_CLEARS) | (kind == TOOL_TRANSFORMS)
    idx = jnp.where(acts, encodings[:, 1], num_waste_types).astype(jnp.int32)
    tools = jnp.full((num_waste_types + 1,), -1, dtype=jnp.int16)
    tools = tools.at[idx].set(encodings[:, 2].astype(jnp.int16))
    nxt_val = jnp.where(kind == TOOL_TRANSFORMS,
                        encodings[:, 3].astype(jnp.int16), jnp.int16(-1))
    nexts = jnp.full((num_waste_types + 1,), -1, dtype=jnp.int16)
    nexts = nexts.at[idx].set(nxt_val)
    a_tools, a_nexts = decode_attr_chain(encodings, num_waste_types, num_shades)
    attr = is_attr_ruleset(encodings)
    return (jnp.where(attr, a_tools, tools[:num_waste_types]),
            jnp.where(attr, a_nexts, nexts[:num_waste_types]))


def chain_depth_of(encodings: jax.Array, num_waste_types: int) -> jax.Array:
    """Per colour: how many beam hits until water (1 = direct, 2 = one
    transform then clear, ...). Follows ``nexts`` for at most MAX_RULES
    hops, so a malformed cycle cannot hang the trace."""
    tools, nexts = decode_chain(encodings, num_waste_types)
    k = jnp.arange(num_waste_types)
    depth = jnp.where(tools >= 0, 1, 0)
    cur = nexts[k]
    for _ in range(MAX_RULES):
        more = cur >= 0
        depth = depth + more.astype(depth.dtype)
        cur = jnp.where(more, nexts[jnp.clip(cur, 0, num_waste_types - 1)], -1)
    return depth


# ---------------------------------------------------------------------------
# open_harvest rulesets: per hue a regrowth-rate class and a pay order over the ripeness stages
# ---------------------------------------------------------------------------
REGROW_RATES = (2.0, 1.0, 0.5)         # fast / normal / slow multiplier on the vanilla neighbour-count regrowth probabilities (2026-09-19: slow 0.25 -> 0.5, user; 09-16: fast 4 -> 2, no 'never')
NUM_REGROW_RATES = len(REGROW_RATES)
NUM_RIPE_STAGES = 3                     # green, ripe, over-ripe
NUM_HARVEST_HUES = 5
# Every hue pays the same three amounts, one per ripeness stage, in a hidden order (2026-09-16):
# PAY_ORDERS[o][s] is the pay rank of stage s under order o, and STAGE_PAY[rank] the points.
STAGE_PAY = (1.0, 0.5, 0.25)
PAY_ORDERS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))   # itertools.permutations(range(3))
NUM_PAY_ORDERS = len(PAY_ORDERS)
PAY_TABLE = jnp.array([[STAGE_PAY[r] for r in o] for o in PAY_ORDERS], dtype=jnp.float32)   # (order, stage) -> points
PEAK_OF_ORDER = jnp.array([o.index(0) for o in PAY_ORDERS], dtype=jnp.int32)               # the stage that pays 1
DEFAULT_PAY_ORDER = 2                   # (1, 0, 2): ripe pays 1, unripe 0.5, over-ripe 0.25 -- the fallback for a hue without a row
# The regrowth speed of an empty cell depends on the stage its last apple had when it was eaten or rotted, again in a
# hidden per-hue order: RATE_TABLE[order, stage] = REGROW_RATES[PAY_ORDERS[order][stage]] (2026-09-16).
RATE_TABLE = jnp.array([[REGROW_RATES[r] for r in o] for o in PAY_ORDERS], dtype=jnp.float32)   # (speed order, stage) -> multiplier
DEFAULT_RATE_ORDER = 2                  # (1, 0, 2): a cell emptied at ripe regrows fast, at unripe normal, at over-ripe slow


def encode_harvest_ruleset(rate_idx, order, max_rules: int = MAX_RULES) -> jax.Array:
    """``rate_idx[h]`` = speed order and ``order[h]`` = pay order of hue h, both in [0, NUM_PAY_ORDERS)
    -> padded ``(max_rules, RULE_ENCODING_LEN)`` encoding: one REGROW_RATE (speed-order) row and one
    RIPE_PEAK (pay-order) row per hue. jit/vmap-friendly."""
    rate_idx = jnp.asarray(rate_idx, dtype=jnp.uint8); peak = jnp.asarray(order, dtype=jnp.uint8)
    n = rate_idx.shape[0]
    hues = jnp.arange(n, dtype=jnp.uint8); zero = jnp.zeros(n, dtype=jnp.uint8)
    rate_rows = jnp.stack([jnp.full(n, REGROW_RATE, dtype=jnp.uint8), hues, rate_idx, zero], axis=1)
    peak_rows = jnp.stack([jnp.full(n, RIPE_PEAK, dtype=jnp.uint8), hues, peak, zero], axis=1)
    return pad_ruleset(jnp.concatenate([rate_rows, peak_rows], axis=0), max_rules)


def decode_harvest_ruleset(encodings: jax.Array, num_hues: int = NUM_HARVEST_HUES):
    """``(speed_order (num_hues,), pay_order (num_hues,))`` as int32. A hue without a row
    falls back to DEFAULT_RATE_ORDER and DEFAULT_PAY_ORDER."""
    ids = encodings[:, 0].astype(jnp.int32); hue = encodings[:, 1].astype(jnp.int32); arg = encodings[:, 2].astype(jnp.int32)
    hues = jnp.arange(num_hues)
    is_rate = (ids == REGROW_RATE)[None, :] & (hue[None, :] == hues[:, None])
    is_peak = (ids == RIPE_PEAK)[None, :] & (hue[None, :] == hues[:, None])
    rate_idx = jnp.where(is_rate.any(axis=1), (is_rate * arg[None, :]).sum(axis=1), DEFAULT_RATE_ORDER)
    order = jnp.where(is_peak.any(axis=1), (is_peak * arg[None, :]).sum(axis=1), DEFAULT_PAY_ORDER)
    return rate_idx, order


def num_harvest_rulesets(num_hues: int = NUM_HARVEST_HUES) -> int:
    return (NUM_PAY_ORDERS * NUM_PAY_ORDERS) ** num_hues     # (6 speed orders x 6 pay orders) per hue = 36**5 = 60,466,176


def harvest_rulesets_from_index(idx: jax.Array, num_hues: int = NUM_HARVEST_HUES) -> jax.Array:
    """Ruleset number ``idx`` in [0, num_harvest_rulesets()) -> encoding; base-36 digits, one per hue (speed order x pay order)."""
    per_hue = NUM_PAY_ORDERS * NUM_PAY_ORDERS
    digits = jnp.stack([(idx // per_hue ** h) % per_hue for h in range(num_hues)], axis=1)   # (N, num_hues)
    rate_idx = digits // NUM_PAY_ORDERS; order = digits % NUM_PAY_ORDERS
    return jax.vmap(encode_harvest_ruleset)(rate_idx.astype(jnp.uint8), order.astype(jnp.uint8))


def sampled_harvest_rulesets(n: int = 100_000, seed: int = 0, num_hues: int = NUM_HARVEST_HUES) -> jax.Array:
    """``n`` distinct rulesets drawn from the 36**5 = 60,466,176 space (too many to enumerate as
    encodings); fixed ``seed`` -> the same benchmark every time (numpy draw, duplicates dropped)."""
    import numpy as _np
    total = num_harvest_rulesets(num_hues)
    idx = _np.unique(_np.random.default_rng(seed).integers(0, total, size=n + 1000))
    assert len(idx) >= n, "unexpectedly many duplicate draws"
    return harvest_rulesets_from_index(jnp.asarray(idx[:n]), num_hues)


def all_harvest_rulesets(num_hues: int = NUM_HARVEST_HUES) -> jax.Array:
    """Every ruleset, enumerated: 60,466,176 x (MAX_RULES, RULE_ENCODING_LEN) -- ~4 GB; use sampled_harvest_rulesets instead."""
    return harvest_rulesets_from_index(jnp.arange(num_harvest_rulesets(num_hues)), num_hues)


def sample_harvest_ruleset(key, num_hues: int = NUM_HARVEST_HUES) -> jax.Array:
    k1, k2 = jax.random.split(key)
    return encode_harvest_ruleset(jax.random.randint(k1, (num_hues,), 0, NUM_PAY_ORDERS),
                                  jax.random.randint(k2, (num_hues,), 0, NUM_PAY_ORDERS))


def is_harvest_ruleset(encodings: jax.Array) -> jax.Array:
    return jnp.any(encodings[..., 0] == REGROW_RATE, axis=-1)
