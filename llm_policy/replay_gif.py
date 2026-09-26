"""Rebuild the GIF of an open_cleanup_policy run from its decisions log.

The environment is deterministic given the seed and the key-splitting order of
run_episode(), so replaying the logged actions reproduces the exact states;
positions are checked against the log every step.  Writes a GIF for every
trial in --gif-trials, including a trial cut short by a crash.

usage: python llm_policy/replay_gif.py --log logs/llm/v2h_gpt55_decisions.jsonl --tag v2h_gpt55 \
           --seed 906 --outer 1 --inner 200 --gif-dir logs/llm/gifs_v2h_gpt55
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import jax
import numpy as onp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_policy.open_cleanup_policy import FIRE, RAW_TO_ENV, curriculum_start, make_env  # noqa: E402
from llm_policy.record import GifRecorder  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--gif-dir", required=True); ap.add_argument("--gif-trials", default="0")
    ap.add_argument("--gif-every", type=int, default=1); ap.add_argument("--gif-fps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=906); ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--inner", type=int, default=200); ap.add_argument("--outer", type=int, default=4)
    ap.add_argument("--spawn", type=float, default=0.1); ap.add_argument("--skew", type=float, default=0.8)
    ap.add_argument("--on-train", action="store_true"); ap.add_argument("--curriculum", default="none")
    ap.add_argument("--random-layout", action="store_true", help="the run played on a random map (same pool and seed)")
    a = ap.parse_args()

    by_step = defaultdict(dict)
    for line in open(a.log):
        r = json.loads(line)
        if "outcome" not in r:                      # trial-boundary notebook records carry no step
            continue
        if r["episode"] == a.episode:
            by_step[(r["trial"], r["t"])][r["agent"]] = r
    steps = sorted(by_step)
    print(f"{len(steps)} logged steps, trials {sorted(set(s[0] for s in steps))}", flush=True)

    env = make_env(a.agents, a.inner, a.outer, a.spawn, a.skew, a.on_train,
                   layouts="random" if a.random_layout else "fixed")
    n = env.num_agents
    key = jax.random.PRNGKey(a.seed + a.episode)
    key, kr, ks = jax.random.split(key, 3)
    _, state = env.reset(kr)
    state = curriculum_start(env, state, a.curriculum)
    step = jax.jit(env.step)
    own_scale = 1.0 / n
    rec = GifRecorder(env, a.gif_dir, a.tag, trials=[int(x) for x in a.gif_trials.split(",")], every=a.gif_every, fps=a.gif_fps)

    mismatches = 0
    for (trial, t) in steps:
        assert (int(state.outer_t), int(state.inner_t)) == (trial, t), f"state at {(int(state.outer_t), int(state.inner_t))}, log at {(trial, t)}"
        recs = by_step[(trial, t)]
        if len(recs) != n:
            print(f"step {(trial, t)}: only {len(recs)} of {n} agents logged, stopping here", flush=True)
            break
        raw_acts = [recs[i]["outcome"]["action"] for i in range(n)]
        ks, k1 = jax.random.split(ks)
        _, state, rew, dones, info = step(k1, state, [RAW_TO_ENV[x] for x in raw_acts])
        own = onp.asarray(info.get("original_rewards", rew)).reshape(-1) * own_scale
        locs = onp.asarray(state.agent_locs)
        for i in range(n):
            if [int(x) for x in locs[i][:2]] != recs[i]["outcome"]["moved_to"]:
                mismatches += 1
        trial_ended = int(state.inner_t) == 0 and int(state.outer_t) != trial
        r0 = recs[0]["action"]
        rec.set_caption(f"A0 {r0.get('mode')} {r0.get('action')} - {str(r0.get('reasoning', ''))[:70]}")
        rec(state, {"trial": trial, "t": t, "held": int(onp.asarray(state.held_tool)[0]),
                    "crafted": int(onp.asarray(state.crafted)[0]), "reward": float(own.sum()),
                    "fired_agents": [i for i in range(n) if raw_acts[i] == FIRE]}, trial_ended)
        if bool(dones["__all__"]):
            break
    if rec.frames:
        rec.flush(int(steps[-1][0]) if steps else 0)
    print(f"replayed {len(steps)} steps, position mismatches {mismatches}, wrote {rec.written}", flush=True)


if __name__ == "__main__":
    main()
