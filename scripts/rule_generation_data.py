"""Facts for the rule-generation figures, read from the code (needs JAX; run once, the figures read the JSON):
the seed-906 rule sets of both environments, the benchmark and split sizes, the layout pool.
usage: PYTHONPATH=. python scripts/rule_generation_data.py"""
import json, sys
sys.path.insert(0, ".")
import jax, numpy as onp
from llm_policy import open_cleanup_policy as CP, open_harvest_policy as HP
from opensocialjax.environments.discovery_rules import (decode_harvest_ruleset, PAY_TABLE, RATE_TABLE, REGROW_RATES, STAGE_PAY,
                                                         num_harvest_rulesets, num_attr_craft_rulesets)
from opensocialjax.environments.rule_benchmark import build_benchmark
from opensocialjax.environments.open_cleanup.layouts import default_pool
SEED = 906; out = {"seed": SEED}

# ---- harvest
from opensocialjax.environments.open_harvest.layouts_orig6 import pick_layout
env = HP.make_env(5, 300, 5, 25, False, pick_layout(SEED)); st = HP.initial_state(env, SEED)
speed, pay = (onp.array(x) for x in decode_harvest_ruleset(st.rule_encoding))
word = dict(zip((float(r) for r in REGROW_RATES), ("fast", "normal", "slow")))
b = build_benchmark(harvest=True); tr, te = b.shuffle(jax.random.PRNGKey(0)).split(0.8)
out["harvest"] = {"hues": list(HP.HUE_NAMES) if hasattr(HP, "HUE_NAMES") else None, "drawn": HP.present_hues(st),
                  "pay": [[float(x) for x in onp.array(PAY_TABLE)[o]] for o in pay], "speed": [[word[float(x)] for x in onp.array(RATE_TABLE)[o]] for o in speed],
                  "stage_pay": list(STAGE_PAY), "rates": list(REGROW_RATES), "space": int(num_harvest_rulesets()),
                  "bench": int(b.size), "train": int(tr.size), "held_out": int(te.size)}

# ---- cleanup (the setting of clean38t_rand_0917: shared tools, random layout)
env = CP.make_env(3, 500, 2, 0.1, 0.8, False, layouts="random")
key = jax.random.PRNGKey(SEED); key, kr, ks = jax.random.split(key, 3); _, st = env.reset(kr)
rec = CP.true_recipes(env, st)
g = onp.asarray(st.grid); m = (g >= CP.DIRT_BASE) & (g < CP.DIRT_BASE + CP.NUM_DIRT_TYPES); hues = ((g[m] - CP.DIRT_BASE) // CP.NUM_SHADES).tolist()
b = build_benchmark(chain_depth=1, craft=True, craft_tools=4, craft_len=2, attr=True, attr_tools=4, attr_craft=True); tr, te = b.shuffle(jax.random.PRNGKey(0)).split(0.8)
pool = default_pool(); ptr, pte = pool.split()
out["cleanup"] = {"recipes": {CP.Evidence.short(k): [int(x) for x in v] for k, v in rec.items()}, "dominant_hue": CP.ATTR_HUE_NAMES[max(set(hues), key=hues.count)],
                  "hue_share": round(hues.count(max(set(hues), key=hues.count)) / len(hues), 2), "space": int(num_attr_craft_rulesets(4, 2)),
                  "bench": int(b.size), "train": int(tr.size), "held_out": int(te.size), "layouts": int(pool.size), "layouts_train": int(ptr.size), "layouts_held_out": int(pte.size)}
json.dump(out, open("scripts/rule_generation_data.json", "w"), indent=1); print(json.dumps(out, indent=1))
