"""Held-out rule splits for the two hidden-rule environments.

Each environment enumerates its rule space into a benchmark (rule_benchmark.build_benchmark), shuffles it with
`split_seed` and splits it 80 / 20. Policies train on the first part and are evaluated on the second, so an
evaluation asks whether a policy can infer a rule it has never met, not whether it memorised the training rules.
`on_train=True` scores the training split with the same protocol, which separates "memorised the rules" from
"ignores the rules". The LLM harness (llm_policy) always plays the held-out split.
"""
import jax

import opensocialjax
from opensocialjax.environments.rule_benchmark import build_benchmark


def held_out_env(split=0.8, split_seed=0, chain_depth=1, bijective=False, on_train=False, **env_kwargs):
    """OpenCleanup with rules from the held-out (or training) split, and the river fully polluted at the start
    (initial_dirt=1.0), so no trial scores before the rule matters."""
    bench = build_benchmark(chain_depth=chain_depth, bijective=bijective,
                            craft=bool(env_kwargs.get("craft_rules", False)),
                            craft_tools=int(env_kwargs.get("craft_tools", 4)),
                            craft_len=int(env_kwargs.get("craft_len", 3)),
                            attr=bool(env_kwargs.get("attr_rules", False)),
                            attr_tools=int(env_kwargs.get("attr_tools", 4)),
                            attr_craft=bool(env_kwargs.get("attr_rules", False)) and bool(env_kwargs.get("craft_rules", False)),
                            per_colour=bool(env_kwargs.get("per_colour_tools", False)))
    if bijective:
        env_kwargs["bijective_rules"] = True
    train_rules, test_rules = bench.shuffle(jax.random.PRNGKey(split_seed)).split(split)
    pool = train_rules if on_train else test_rules
    print(f"[{'train split' if on_train else 'held-out'}] {pool.size} rules, initial_dirt=1.0")
    return opensocialjax.make("open_cleanup", rule_pool=pool, initial_dirt=1.0, **env_kwargs)


def held_out_harvest_env(split=0.8, split_seed=0, on_train=False, **env_kwargs):
    """OpenHarvest with rules from the held-out (or training) split of its enumerated benchmark."""
    bench = build_benchmark(harvest=True)
    train_rules, test_rules = bench.shuffle(jax.random.PRNGKey(split_seed)).split(split)
    pool = train_rules if on_train else test_rules
    print(f"[harvest held-out] {bench.size} rulesets -> {train_rules.size} train / {test_rules.size} held out; "
          f"using {'train' if on_train else 'held-out'} ({pool.size})")
    return opensocialjax.make("open_harvest", rule_pool=pool, **env_kwargs)
