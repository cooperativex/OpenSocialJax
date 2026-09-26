"""Paper tables for a finished cleanup formal experiment (default experiments/open_cleanup_5ag).

Per model, over its finished seeds (runs with DONE):
  apples/seed      team apples summed over the 5 trials (mean, sd, per seed)
  T1 / T5 / best   team apples in the first trial, the last trial, the best trial (means over seeds)
  cleared, light   waste cells cleared to water / made lighter by one shade, per seed
  top cleaner      share of the team's cleared cells cleared by its busiest cleaner, mean over trials with any clearing
                   (1/5 = shared evenly, 1 = one agent cleans alone)
  top eater        share of the team's apples eaten by its biggest eater, mean over trials with any apples
  cleaner <= median  trials (with any cleaning and any apples) in which the busiest cleaner ate no more apples than the
                   team's median agent
  zero trials      trials in which the team ate nothing
  invalid          replies that did not parse to a legal action (the agent stays)
  tokens           median output tokens per call; cost per seed (API models)

Only the experiment's shared seeds (setting.seeds) are counted, so every model is scored on the same problems; a
model's extra seeds (e.g. Qwen3.8-27B 909-913) are left out unless --all-seeds is given.

usage: python scripts/summarize_open_cleanup.py [EXP_DIR] [--all-seeds]"""
import glob, json, os, statistics as st, sys

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
EXP = ARGS[0] if ARGS else "experiments/open_cleanup_5ag"
ALL_SEEDS = "--all-seeds" in sys.argv


def tot(x):
    return sum(tot(v) for v in x.values()) if isinstance(x, dict) else (x if isinstance(x, (int, float)) else 0)


cfg = json.load(open(os.path.join(EXP, "config.json")))
SHARED = set(cfg["setting"]["seeds"])
rows = []
for model in cfg["models"]:
    seeds = []
    for d in sorted(glob.glob(os.path.join(EXP, "runs", model, "seed*"))):
        if not os.path.exists(os.path.join(d, "DONE")):
            continue
        r = json.load(open(os.path.join(d, "result.json")))
        if not ALL_SEEDS and r["seed"] not in SHARED:
            continue
        pa = r["per_agent"]                                   # [agent][trial] -> {"return": apples, "cleared": cells}
        T = len(r["team_apples_per_trial"])
        apples = [sum(pa[i][t]["return"] for i in range(len(pa))) for t in range(T)]
        cleared = [sum(pa[i][t]["cleared"] for i in range(len(pa))) for t in range(T)]
        top_clean = [max(pa[i][t]["cleared"] for i in range(len(pa))) / cleared[t] for t in range(T) if cleared[t]]
        top_eat = [max(pa[i][t]["return"] for i in range(len(pa))) / apples[t] for t in range(T) if apples[t]]
        both = [t for t in range(T) if cleared[t] and apples[t]]
        busiest = {t: max(range(len(pa)), key=lambda i: pa[i][t]["cleared"]) for t in both}
        below = sum(1 for t in both if pa[busiest[t]][t]["return"] <= st.median(pa[i][t]["return"] for i in range(len(pa))))
        light = n = inv = 0
        for line in open(os.path.join(d, "decisions.jsonl")):
            x = json.loads(line)
            if "outcome" not in x:
                continue
            o = x["outcome"]; n += 1; inv += bool(o.get("invalid")); light += tot(o.get("transformed") or {})
        toks = []
        for f in glob.glob(os.path.join(d, "calls_agent*.jsonl")):
            for line in open(f):
                c = json.loads(line); toks.append(c.get("tokens") or 0)
        seeds.append(dict(seed=r["seed"], apples=apples, total=sum(apples), cleared=sum(cleared), light=light,
                          top_clean=st.mean(top_clean) if top_clean else None, top_eat=st.mean(top_eat) if top_eat else None,
                          zero=sum(1 for a in apples if a == 0), below=below, both=len(both), inv=inv / max(n, 1), tok=st.median(toks) if toks else None,
                          cost=r.get("cost_usd")))
    if not seeds:
        continue
    m = lambda k: st.mean(s[k] for s in seeds if s[k] is not None)
    tr = lambda t: st.mean(s["apples"][t] for s in seeds)
    rows.append(dict(model=model, n=len(seeds), mean=m("total"), sd=st.stdev([s["total"] for s in seeds]) if len(seeds) > 1 else 0.0,
                     per_seed=[s["total"] for s in seeds], T1=tr(0), T5=tr(-1), best=st.mean(max(s["apples"]) for s in seeds),
                     curve=[round(tr(t)) for t in range(len(seeds[0]["apples"]))], cleared=m("cleared"), light=m("light"),
                     top_clean=m("top_clean"), top_eat=m("top_eat"), zero=sum(s["zero"] for s in seeds), trials=sum(len(s["apples"]) for s in seeds),
                     below=sum(s["below"] for s in seeds), both=sum(s["both"] for s in seeds), inv=m("inv"), tok=m("tok"), cost=(m("cost") if all(s["cost"] is not None for s in seeds) else None)))

rows.sort(key=lambda r: -r["mean"])
print(f"{'model':28s} {'n':>2} {'apples/seed':>16} {'per seed':>22} {'T1':>5} {'T5':>5} {'best':>5} {'cleared':>7} {'light':>6} "
      f"{'topClean':>8} {'topEat':>7} {'zero':>6} {'invalid':>7} {'tok':>6} {'$/seed':>7}")
for r in rows:
    print(f"{r['model']:28s} {r['n']:>2} {r['mean']:>8.0f} ± {r['sd']:<5.0f} {str(r['per_seed']):>22} {r['T1']:>5.0f} {r['T5']:>5.0f} {r['best']:>5.0f} "
          f"{r['cleared']:>7.0f} {r['light']:>6.0f} {r['top_clean']:>8.2f} {r['top_eat']:>7.2f} {r['zero']:>3}/{r['trials']:<2} {r['inv']:>7.1%} "
          f"{r['tok']:>6.0f} {('$%.1f' % r['cost']) if r['cost'] is not None else '-':>7}")
print("\nbusiest cleaner ate no more than the team median:")
for r in rows:
    print(f"  {r['model']:28s} {r['below']}/{r['both']} trials")
print("\nmean team apples per trial (T1..T5):")
for r in rows:
    print(f"  {r['model']:28s} {r['curve']}")
