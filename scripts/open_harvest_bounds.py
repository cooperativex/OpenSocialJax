"""Scripted upper / lower bounds for harvest on the harness baseline: the harness env (pool map of the seed,
5 agents, 5 x 300, respawn 12), the harness's reset key for seed 906, so the same map, colours and hidden rules."""
import os, sys, jax, numpy as onp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_policy.open_harvest_policy import make_env
from opensocialjax.environments.open_harvest.layouts_orig6 import pick_layout
from tests.harvest_gate import run
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 906
env = make_env(5, 300, 5, layout=pick_layout(SEED))
_, key, _ = jax.random.split(jax.random.PRNGKey(SEED), 3)          # open_harvest_policy.initial_state / run_episode reset key
for kind, keep, leave in [("greedy", 3, 0.0), ("patient", 3, 0.0), ("oracle", 3, 0.0), ("oracle", 2, 0.0), ("oracle_opt", 3, 0.0), ("oracle_fast", 3, 0.0)]:
    per, (pay, speed), peaks, _ = run(kind, env, key, 5, 300, keep, leave=leave)
    tot = [round(sum(p["score"]), 1) for p in per]
    print(f"{kind:11s} keep={keep} | team points/trial {tot} total {round(sum(tot), 1)} | eaten/trial {[p['eaten'] for p in per]} | rotted {[p['rotted'] for p in per]}", flush=True)
