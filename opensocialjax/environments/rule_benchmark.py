"""Pre-generated rule benchmarks with a train/test split, XLand style.

XLand-MiniGrid does not sample tasks from the whole space during training. It
pre-generates a fixed set of rulesets **offline**, pads them to one shape,
stacks them into arrays, and ships those arrays; training then only ever
*indexes* into them, which is trivially jit/vmap-compatible. This module is
that construction for the hidden-rule environments (open_cleanup, open_harvest):

- a ruleset is a padded ``(MAX_RULES, RULE_ENCODING_LEN)`` uint8 encoding
  (see ``discovery_rules`` — zero rows are the EmptyRule sentinel);
- a benchmark stacks N of them, ``(N, MAX_RULES, RULE_ENCODING_LEN)``, next
  to a ``num_rules`` count per ruleset (the logical size the padding hides);
- generation — enumeration, sampling, dedup — happens once, host-side, in
  plain Python, and can be saved to disk; nothing non-jittable survives into
  the training step.

The split is the part that carries the science. Training on one subset and
evaluating on the held-out rest is the only thing separating *inferring a
rule online* from *memorising every rule into the weights*. Two ways to cut:

``shuffle(key).split(prop)``
    random — the held-out rules are new tool tables, but every (waste, tool)
    pairing they use has been seen inside other training rules.

``filter_split(fn)``
    by predicate on the encoding — e.g. "every ruleset where waste type 0 is
    cleared by tool 2 is held out". The test set is then *semantically*
    unseen: that pairing never occurs in training, so scoring on it requires
    reading evidence, not recognising a neighbour. This mirrors XLand's
    ``Benchmark.filter_split``.

Usage::

    bench = enumerate_benchmark(all_rulesets())          # open_cleanup
    train, test = bench.shuffle(key).split(0.8)          # random split
    train, test = bench.filter_split(
        lambda enc: decode_tool_table(enc, 6)[0] != 2)   # semantic split
    ruleset = train.sample(key)      # one (MAX_RULES, ENC_LEN) encoding

Nothing here touches environment mechanics; it only decides which ruleset an
episode is given.
"""

import bz2
import pickle

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from opensocialjax.environments.discovery_rules import count_rules


@dataclass
class RuleBenchmark:
    """A fixed collection of ruleset encodings.

    ``rules``     — ``(N, MAX_RULES, RULE_ENCODING_LEN)`` uint8
    ``num_rules`` — ``(N,)`` uint8, non-Empty rows per ruleset
    """

    rules: jnp.ndarray
    num_rules: jnp.ndarray

    @property
    def size(self):
        return self.rules.shape[0]

    def get_ruleset(self, ruleset_id):
        """Index one ruleset; jit-safe for traced ids."""
        return jax.lax.dynamic_index_in_dim(self.rules, ruleset_id, keepdims=False)

    def sample(self, key):
        """Draw one ruleset uniformly. Mirrors XLand's ``sample_ruleset`` —
        under ``vmap`` over keys, every parallel environment gets its own."""
        idx = jax.random.randint(key, (), 0, self.rules.shape[0])
        return self.get_ruleset(idx)

    def shuffle(self, key):
        idxs = jax.random.permutation(key, jnp.arange(self.size))
        return jax.tree.map(lambda a: a[idxs], self)

    def split(self, prop=0.8):
        """Split into (train, test) at ``prop``. Shuffle first — generators
        emit rulesets in a structured order, so an unshuffled split would put
        whole regions of the space in one half."""
        idx = round(self.size * prop)
        return (jax.tree.map(lambda a: a[:idx], self),
                jax.tree.map(lambda a: a[idx:], self))

    def filter_split(self, fn):
        """Split by predicate: ``fn(encoding) -> bool`` over one ruleset's
        ``(MAX_RULES, RULE_ENCODING_LEN)`` encoding. True goes to the first
        benchmark (train), False to the second (held out) — XLand's
        ``Benchmark.filter_split``. Host-side, once, at setup."""
        mask = jax.vmap(fn)(self.rules)
        return (jax.tree.map(lambda a: a[mask], self),
                jax.tree.map(lambda a: a[~mask], self))


def enumerate_benchmark(encoded_rulesets):
    """Build a benchmark from an explicit ``(N, MAX_RULES, ENC_LEN)`` stack of
    ruleset encodings.

    Preferred when the space is small enough to enumerate exhaustively:
    sampling would need many draws to cover it and would still miss rulesets.
    """
    rules = jnp.asarray(encoded_rulesets, dtype=jnp.uint8)
    return RuleBenchmark(rules=rules, num_rules=count_rules(rules))


def generate_benchmark(key, sample_ruleset_fn, num_rulesets, dedup=True):
    """Draw ``num_rulesets`` encodings with ``sample_ruleset_fn(key)``.

    The offline generator half of the XLand recipe: run once at setup (or in
    a script that saves the result), never inside a training step. With
    ``dedup`` only distinct encodings are kept, so a later train/test split
    really is disjoint — sampling with replacement from a small space would
    otherwise leak the same ruleset into both halves and quietly turn the
    held-out set into a training set.
    """
    keys = jax.random.split(key, num_rulesets)
    rules = jax.vmap(sample_ruleset_fn)(keys)
    if dedup:
        # host-side: this runs once at setup, not inside a training step
        import numpy as onp
        flat = onp.asarray(rules).reshape(rules.shape[0], -1)
        _, keep = onp.unique(flat, axis=0, return_index=True)
        rules = jnp.asarray(rules)[jnp.sort(jnp.asarray(keep))]
    return enumerate_benchmark(rules)


def save_benchmark(benchmark, path):
    """Persist a benchmark the way XLand ships its own: arrays in a
    bz2-compressed pickle, so a generated benchmark is a file, not a seed."""
    payload = {
        "rules": jax.device_get(benchmark.rules),
        "num_rules": jax.device_get(benchmark.num_rules),
    }
    with bz2.open(path, "wb") as f:
        pickle.dump(payload, f, protocol=-1)


def load_benchmark(path):
    with bz2.open(path, "rb") as f:
        payload = pickle.load(f)
    return RuleBenchmark(
        rules=jnp.asarray(payload["rules"], dtype=jnp.uint8),
        num_rules=jnp.asarray(payload["num_rules"], dtype=jnp.uint8),
    )


def split_from_config(bench, config, num_waste_types):
    """The train/test cut both training entry points share, so the recurrent
    run and its memoryless control always price the same distribution.

    ``RULE_SPLIT_MODE: random`` (default) — ``shuffle(RULE_SPLIT_SEED)`` then
    ``split(RULE_SPLIT)``, the historical behaviour.

    ``RULE_SPLIT_MODE: predicate`` — hold out every ruleset where waste type
    ``RULE_HOLDOUT_WASTE`` is cleared by tool ``RULE_HOLDOUT_TOOL``. That
    (waste, tool) pairing then never occurs in training, so the held-out set
    is semantically unseen, not just unseen-by-index.
    """
    from opensocialjax.environments.discovery_rules import decode_tool_table

    mode = config.get("RULE_SPLIT_MODE", "random")
    if mode == "predicate":
        waste = int(config.get("RULE_HOLDOUT_WASTE", 0))
        tool = int(config.get("RULE_HOLDOUT_TOOL", 2))
        train, test = bench.filter_split(
            lambda enc: decode_tool_table(enc, num_waste_types)[waste] != tool
        )
        desc = f"predicate: waste {waste} cleared by tool {tool} held out"
    elif mode == "random":
        train, test = bench.shuffle(
            jax.random.PRNGKey(config.get("RULE_SPLIT_SEED", 0))
        ).split(config.get("RULE_SPLIT", 0.8))
        desc = f"random, seed {config.get('RULE_SPLIT_SEED', 0)}"
    else:
        raise ValueError(
            f"Unknown RULE_SPLIT_MODE {mode!r}; use 'random' or 'predicate'"
        )
    print(f"[rule benchmark] {bench.size} rulesets -> "
          f"{train.size} train / {test.size} held out ({desc})")
    return train, test


def build_benchmark(chain_depth=1, p_chain=0.5, bench_seed=0, bench_size=100_000,
                    bijective=False, craft=False, craft_tools=4, craft_len=3,
                    attr=False, attr_tools=4, attr_craft=False, harvest=False, per_colour=False):
    """The benchmark every entry point and analyzer agrees on.

    ``attr=True``: the attribute-chain space, enumerated (12288 rulesets at
    four tools). Only ``RULE_SPLIT_MODE: random`` is meaningful for it — the
    predicate split reads ``decode_tool_table``, which is blind to its rows.
    ``chain_depth=1``: the tool-table space (3**15 tables, sampled).
    ``chain_depth>=2``: XLand's offline recipe — sample ``bench_size`` chain
    rulesets with ``sample_chain_ruleset`` under a fixed seed, dedup. Fixed
    seed + size make the sampled benchmark reproducible, so training and
    later evaluation see the same split.
    """
    from functools import partial
    if harvest:
        # open_harvest: (regrowth-speed order x pay order over the 3 stages) per hue, 36**5 = 60,466,176
        # rulesets; 100,000 sampled with a fixed seed (like the cleanup attr-craft benchmark)
        from opensocialjax.environments.discovery_rules import sampled_harvest_rulesets
        return enumerate_benchmark(sampled_harvest_rulesets())
    from opensocialjax.environments.open_cleanup.open_cleanup import (
        ENUMERABLE_LIMIT, all_rulesets, rule_space_size, sample_bijective_ruleset,
        sample_chain_ruleset, sample_ruleset,
    )
    if attr_craft and per_colour:
        # one recipe per waste colour: P(craft_tools**craft_len, 15) rulesets, sampled
        from opensocialjax.environments.discovery_rules import sample_type_craft_ruleset
        return generate_benchmark(jax.random.PRNGKey(bench_seed),
                                  partial(sample_type_craft_ruleset, num_tools=craft_tools, k=craft_len),
                                  bench_size, dedup=True)
    if attr_craft:
        # every acting tool crafted: P(craft_tools**craft_len, 7) rulesets, sampled
        from opensocialjax.environments.discovery_rules import sample_attr_craft_ruleset
        from opensocialjax.environments.open_cleanup.open_cleanup import NUM_HUES, NUM_SHADES
        return generate_benchmark(
            jax.random.PRNGKey(bench_seed),
            partial(sample_attr_craft_ruleset, num_tools=craft_tools, k=craft_len,
                    num_hues=NUM_HUES, num_shades=NUM_SHADES),
            bench_size, dedup=True)
    if attr:
        from opensocialjax.environments.discovery_rules import all_attr_rulesets
        from opensocialjax.environments.open_cleanup.open_cleanup import NUM_HUES, NUM_SHADES
        return enumerate_benchmark(all_attr_rulesets(attr_tools, NUM_HUES, NUM_SHADES))
    if craft:
        # the crafting variant: every pickup order, craft_tools ** craft_len rows
        from opensocialjax.environments.discovery_rules import all_craft_rulesets
        return enumerate_benchmark(all_craft_rulesets(craft_tools, craft_len))
    if bijective:
        # permutations: 10! ~ 3.6M, sampled like the chain space
        return generate_benchmark(jax.random.PRNGKey(bench_seed),
                                  sample_bijective_ruleset, bench_size, dedup=True)
    if chain_depth >= 2:
        return generate_benchmark(
            jax.random.PRNGKey(bench_seed),
            partial(sample_chain_ruleset, depth=chain_depth, p_chain=p_chain),
            bench_size, dedup=True)
    if rule_space_size() > ENUMERABLE_LIMIT:
        # too many tables to enumerate: XLand's recipe for big spaces —
        # sample, dedup, ship the arrays
        return generate_benchmark(jax.random.PRNGKey(bench_seed),
                                  sample_ruleset, bench_size, dedup=True)
    return enumerate_benchmark(all_rulesets())
