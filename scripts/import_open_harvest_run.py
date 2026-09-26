"""Import a finished stand-alone open_harvest_policy.py run into a formal Harvest experiment as one (model, seed).

The run must have been played with exactly the experiment's setting; the script checks every setting it can read from
the run's args (agents, trials, steps, history, respawn wait, ripening / rot / regrowth rules, seed, map pool) and
refuses otherwise. It writes runs/<model>/seed<k>/ with result.json in the runner's format, the run's decisions.jsonl
and system.txt, an IMPORTED note naming the source, and DONE. The source files are left as they are.

usage: python scripts/import_open_harvest_run.py <EXP_DIR> <MODEL> <TAG>      e.g. ... experiments/open_harvest_5ag gpt-6-sol hv6_sol5_0924
"""
import json, os, shutil, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from llm_policy.run_experiment import cost_of, load_config, run_dir

exp, model, tag = sys.argv[1:4]
cfg = load_config(exp); s = cfg["setting"]; spec = cfg["models"][model]
src = json.load(open(f"logs/llm/llm_{tag}.json")); a = src["args"]; r = src["results"][0]
checks = {"agents": (a["agents"], s["agents"]), "outer": (a["outer"], s["outer"]), "inner": (a["inner"], s["inner"]),
          "history": (a["history"], s["history"]), "respawn_wait": (a["respawn_wait"], s["respawn_wait"]),
          "ripen": (a["ripen"], s["ripen"]), "ripen_range": (a["ripen_range"], s["ripen_range"]),
          "rot_range": (a["rot_range"], s["rot_range"]), "rates": (a["rates"], s["rates"]),
          "early_stop": (a.get("early_stop", True), s["early_stop"]), "pool": (r.get("pool"), s["pool"]),
          "reveal_rules": (a.get("reveal_rules", False), False), "reveal_regrowth": (a.get("reveal_regrowth", False), False)}
bad = {k: v for k, v in checks.items() if v[0] != v[1]}
assert not bad, f"the run's setting differs from the experiment's: {bad}"
assert a["seed"] in s["seeds"], (a["seed"], s["seeds"])
seed = a["seed"]; d = run_dir(exp, model, seed)
assert not os.path.exists(os.path.join(d, "DONE")), f"{d} is already DONE"
os.makedirs(d, exist_ok=True)
usage = {}
for u in r["usage"]:
    for k, v in u.items():
        if isinstance(v, (int, float)):
            usage[k] = usage.get(k, 0) + v
per_agent = r["per_agent"]
res = {"model": model, "seed": seed, "pool": r["pool"], "layout": r["layout"], "team_points_per_trial": r["team"],
       "team_apples_per_trial": [sum(pa[t]["apples"] for pa in per_agent) for t in range(a["outer"])], "per_agent": per_agent,
       "truth": r["truth"], "notebook": r["notebook"], "usage": usage, "cost_usd": cost_of(usage, spec.get("price")),
       "imported_from": f"logs/llm/llm_{tag}.json"}
json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1)
json.dump({"model": model, "seed": seed, "pool": r["pool"], "layout": r["layout"], "spec": spec, "setting": s,
           "imported_from": tag, "source_args": a}, open(os.path.join(d, "run_config.json"), "w"), indent=1)
shutil.copy(f"logs/llm/{tag}_decisions.jsonl", os.path.join(d, "decisions.jsonl"))
shutil.copy(f"logs/llm/{tag}_system.txt", os.path.join(d, "system.txt"))
open(os.path.join(d, "IMPORTED"), "w").write(f"imported from logs/llm/{tag}_* on {time.strftime('%Y-%m-%d %H:%M:%S')} (same setting)\n")
open(os.path.join(d, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
print(f"{model} seed {seed}: imported {tag} | team points/trial {[round(x, 1) for x in r['team']]} | cost ${res['cost_usd']}")
