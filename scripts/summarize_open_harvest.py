"""Paper tables for a finished Harvest formal experiment (default experiments/open_harvest_5ag).

Per model, over its finished runs on the experiment's shared seeds:
  points/seed    team points summed over the 5 trials (mean, sd, per seed)
  position       (points - greedy) / (oracle - greedy) on the same seed, where greedy = every agent eats the nearest
                 apple, oracle = the best scripted full-information policy (knows the hidden rules, eats only at the
                 stage that pays most, keeps >= 3 apples around every emptied cell); from bounds/seed<k>.txt, written by
                 scripts/open_harvest_bounds.py <seed>. 0 = no better than grabbing, 1 = the cooperative optimum.
  T1 / T5        mean team points in the first / last trial
  best stage     share of eaten apples taken at the ripeness stage that pays most for their colour (first -> last trial)
  alive until    mean step at which the orchard was dead (no apple left and none can regrow), 300 = never within the trial
  regrowth       apples eaten beyond the 64 trees, per trial
  zaps           beam shots per trial
  invalid        replies that did not parse to a legal action

usage: python scripts/summarize_open_harvest.py [EXP_DIR]"""
import glob, json, os, re, statistics as st, sys

EXP = sys.argv[1] if len(sys.argv) > 1 else "experiments/open_harvest_5ag"
cfg = json.load(open(os.path.join(EXP, "config.json")))
SEEDS = cfg["setting"]["seeds"]; INNER = cfg["setting"]["inner"]; TREES = 64


def bounds(seed):
    txt = open(os.path.join(EXP, "bounds", f"seed{seed}.txt")).read()
    tot = {}
    for line in txt.splitlines():
        m = re.match(r"(\w+)\s+keep=(\d)\s.*total ([\d.]+)", line)
        if m:
            tot[(m.group(1), int(m.group(2)))] = float(m.group(3))
    return tot[("greedy", 3)], max(v for (k, keep), v in tot.items() if k.startswith("oracle"))


B = {s: bounds(s) for s in SEEDS}
rows = []
for model in cfg["models"]:
    runs = []
    for s in SEEDS:
        d = os.path.join(EXP, "runs", model, f"seed{s}")
        if not os.path.exists(os.path.join(d, "DONE")):
            continue
        r = json.load(open(os.path.join(d, "result.json")))
        T = len(r["team_points_per_trial"])
        eats = {t: [] for t in range(T)}; zaps = [0] * T; dead = {}; n = inv = 0
        for line in open(os.path.join(d, "decisions.jsonl")):
            x = json.loads(line)
            if x.get("event") == "fast_forward":
                dead[int(x["trial"])] = int(x["from_t"]); continue
            o = x.get("outcome"); a = x.get("action")
            if not isinstance(o, dict):
                continue
            t = int(x["trial"]); n += 1
            inv += bool(isinstance(a, dict) and a.get("invalid"))
            if o.get("ate"):
                eats[t].append(o["points"])
            if o.get("action") == 6:
                zaps[t] += 1
        total = sum(r["team_points_per_trial"]); g, o_ = B[s]
        best = [sum(1 for p in eats[t] if p >= 0.99) / len(eats[t]) if eats[t] else 0.0 for t in range(T)]
        runs.append(dict(seed=s, total=total, pos=(total - g) / (o_ - g), T1=r["team_points_per_trial"][0], T5=r["team_points_per_trial"][-1],
                         best1=best[0], best5=best[-1], alive=st.mean(dead.get(t, INNER) for t in range(T)),
                         regrow=st.mean(max(0, a - TREES) for a in r["team_apples_per_trial"]), zaps=st.mean(zaps),
                         inv=inv / max(n, 1), cost=r.get("cost_usd")))
    if not runs:
        continue
    m = lambda k: st.mean(x[k] for x in runs)
    rows.append(dict(model=model, n=len(runs), total=m("total"), sd=st.stdev([x["total"] for x in runs]) if len(runs) > 1 else 0.0,
                     per_seed=[round(x["total"], 1) for x in runs], pos=m("pos"), pos_seeds=[round(x["pos"], 2) for x in runs],
                     T1=m("T1"), T5=m("T5"), best1=m("best1"), best5=m("best5"), alive=m("alive"), regrow=m("regrow"),
                     zaps=m("zaps"), inv=m("inv"), cost=m("cost") if all(x["cost"] is not None for x in runs) else None))

rows.sort(key=lambda r: -r["pos"])
print("bounds (greedy / oracle, 5-trial team points):", {s: (round(g, 1), round(o, 1)) for s, (g, o) in B.items()})
print(f"{'model':28s} {'n':>2} {'points/seed':>14} {'per seed':>22} {'position':>8} {'per seed':>18} {'T1':>5} {'T5':>5} "
      f"{'best T1->T5':>12} {'alive':>6} {'regrow':>6} {'zaps':>5} {'invalid':>7} {'$/seed':>7}")
for r in rows:
    print(f"{r['model']:28s} {r['n']:>2} {r['total']:>7.1f} ± {r['sd']:<5.1f} {str(r['per_seed']):>22} {r['pos']:>8.2f} {str(r['pos_seeds']):>18} "
          f"{r['T1']:>5.1f} {r['T5']:>5.1f} {r['best1']:>5.2f}->{r['best5']:<5.2f} {r['alive']:>6.0f} {r['regrow']:>6.1f} {r['zaps']:>5.1f} "
          f"{r['inv']:>7.1%} {('$%.1f' % r['cost']) if r['cost'] is not None else '-':>7}")
