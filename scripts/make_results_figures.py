"""Paper figures for the homogeneous-population experiments (experiments/open_cleanup_5ag, experiments/open_harvest_5ag):

  docs/analysis/results_trials.{pdf,png}   team return per trial, mean over runs with a +-1 s.d. band, per model;
                                                OpenHarvest also shows the scripted greedy and oracle references
  docs/analysis/results_models.{pdf,png}   one point per run and the mean +- 1 s.d. per model: OpenCleanup team
                                                apples per seed; OpenHarvest normalised score (R - greedy) / (oracle - greedy)

Runs: every finished run of the shared seeds (setting.seeds). Qwen3.8-27B also has two replicates of each OpenCleanup
seed (same problem, other run randomness: seed906r1 ...); they are included for it, so its OpenCleanup statistics are
over 9 runs, and they are drawn hollow in the model figure.

usage: python scripts/make_results_figures.py"""
import glob, json, os, re, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "docs/analysis"
NAME = {"claude-opus-5.5": "Claude Opus 5.5", "gemma-4-31b": "Gemma 4 31B", "gpt-6-luna": "gpt-6-luna", "glm-5.3-flash": "GLM-5.3-Flash",
        "gpt-6-sol": "gpt-6-sol", "qwen3.8-27b": "Qwen3.8-27B", "deepseek-v4.1-flash-nothink": "DeepSeek V4.1 Flash", "qwen3.5-4b": "Qwen3.5-4B"}
COL = {"claude-opus-5.5": "#D96B00", "gemma-4-31b": "#C0392B", "gpt-6-luna": "#5DADE2", "glm-5.3-flash": "#8E5BC9",
       "gpt-6-sol": "#1F5FBF", "qwen3.8-27b": "#3FA64B", "deepseek-v4.1-flash-nothink": "#17A2A2", "qwen3.5-4b": "#8C8C8C"}
MODELS = list(NAME)


def runs(exp, model, key, replicates=False):
    """[(run name, per-trial list)] for the finished runs of the shared seeds (and, if asked, their replicates)."""
    cfg = json.load(open(os.path.join(exp, "config.json"))); seeds = [str(s) for s in cfg["setting"]["seeds"]]
    out = []
    for d in sorted(glob.glob(os.path.join(exp, "runs", model, "seed*"))):
        name = os.path.basename(d)[4:]
        base = name.split("r")[0]
        if base not in seeds or (name != base and not replicates) or not os.path.exists(os.path.join(d, "DONE")):
            continue
        out.append((name, json.load(open(os.path.join(d, "result.json")))[key]))
    return out


def bounds(exp, seed):
    """Per-trial greedy and oracle team points on a seed (bounds/seed<k>.txt, keep=3 rows; oracle = best variant)."""
    rows = {}
    for line in open(os.path.join(exp, "bounds", f"seed{seed}.txt")):
        m = re.match(r"(\w+)\s+keep=3 \| team points/trial \[([^\]]+)\] total ([\d.]+)", line)
        if m:
            rows[m.group(1)] = ([float(v) for v in m.group(2).split(",")], float(m.group(3)))
    oracle = max((v for k, v in rows.items() if k.startswith("oracle")), key=lambda v: v[1])
    return rows["greedy"], oracle


CL, HV = "experiments/open_cleanup_5ag", "experiments/open_harvest_5ag"
cl = {m: runs(CL, m, "team_apples_per_trial", replicates=(m == "qwen3.8-27b")) for m in MODELS}
hv = {m: runs(HV, m, "team_points_per_trial") for m in MODELS}
hseeds = json.load(open(os.path.join(HV, "config.json")))["setting"]["seeds"]
B = {s: bounds(HV, s) for s in hseeds}
norm = {m: [(n, (sum(v) - B[int(n)][0][1]) / (B[int(n)][1][1] - B[int(n)][0][1])) for n, v in hv[m]] for m in MODELS}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
label = lambda m, env: NAME[m] + (" (9 runs)" if env == "cl" and m == "qwen3.8-27b" else "")
order_cl = sorted(MODELS, key=lambda m: -st.mean(sum(v) for _, v in cl[m]) if cl[m] else 0)

# ---------------------------------------------------------------- figure 1: per-trial curves
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.55))
fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.30, wspace=0.22)
T = [1, 2, 3, 4, 5]
for j, (data, title, ylab) in enumerate([(cl, "(a) OpenCleanup", "team apples per trial"), (hv, "(b) OpenHarvest", "team points per trial")]):
    a = ax[j]
    for m in order_cl:
        if not data[m]:
            continue
        per = [[v[t] for _, v in data[m]] for t in range(5)]
        mu = [st.mean(x) for x in per]; sd = [st.stdev(x) if len(x) > 1 else 0 for x in per]
        a.fill_between(T, [u - s for u, s in zip(mu, sd)], [u + s for u, s in zip(mu, sd)], color=COL[m], alpha=0.10, lw=0)
        a.plot(T, mu, color=COL[m], marker="o", ms=2.8, lw=1.3, label=label(m, "cl" if j == 0 else "hv"))
    if j == 1:
        for k, (lab, ls) in enumerate((("greedy", (0, (4, 2))), ("oracle", (0, (1, 1.5))))):
            ref = [st.mean(B[s][k][0][t] for s in hseeds) for t in range(5)]
            a.plot(T, ref, color="#555555", ls=ls, lw=1.0, label=f"scripted {lab}")
    a.set_xticks(T); a.set_xticklabels([f"T{t}" for t in T]); a.set_title(title, loc="left"); a.set_ylabel(ylab); a.set_ylim(bottom=0)
h, l = ax[1].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="lower center", ncol=5, handlelength=2.0, columnspacing=1.2, bbox_to_anchor=(0.53, 0.0))
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "results_trials.pdf")); fig.savefig(os.path.join(OUT, "results_trials.png"), dpi=220)

# ---------------------------------------------------------------- figure 2: models, one point per run
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.45))
fig.subplots_adjust(left=0.17, right=0.99, top=0.90, bottom=0.17, wspace=0.55)
for j, (vals, title, xlab) in enumerate([({m: [(n, sum(v)) for n, v in cl[m]] for m in MODELS}, "(a) OpenCleanup", "team apples per seed (5 trials)"),
                                         (norm, "(b) OpenHarvest", "normalised score  (0 = greedy, 1 = oracle)")]):
    a = ax[j]; order = sorted([m for m in MODELS if vals[m]], key=lambda m: st.mean(x for _, x in vals[m]))
    for i, m in enumerate(order):
        xs = [x for _, x in vals[m]]; mu = st.mean(xs); sd = st.stdev(xs) if len(xs) > 1 else 0
        a.barh(i, mu, height=0.6, color=COL[m], alpha=0.25, lw=0)
        a.errorbar(mu, i, xerr=sd, fmt="none", ecolor=COL[m], elinewidth=1.1, capsize=2.2)
        for n, x in vals[m]:
            rep = "r" in n
            a.plot(x, i + (0.12 if rep else 0), marker="o", ms=3.4, mfc="white" if rep else COL[m], mec=COL[m], mew=0.9, ls="none")
        a.text(mu, i + 0.36, f"{mu:.0f}" if j == 0 else f"{mu:.2f}", fontsize=5.8, ha="center", va="bottom", color="#333333")
    a.set_yticks(range(len(order))); a.set_yticklabels([label(m, "cl" if j == 0 else "hv") for m in order])
    a.set_title(title, loc="left"); a.set_xlabel(xlab); a.set_xlim(left=0 if j == 0 else min(-0.05, min(x for m in order for _, x in vals[m]) - 0.02))
    if j == 1:
        a.axvline(0, color="#555555", lw=0.8, ls=(0, (4, 2)))
ax[0].legend([Line2D([], [], marker="o", ms=3.4, color="#555555", ls="none"), Line2D([], [], marker="o", ms=3.4, mfc="white", mec="#555555", ls="none")],
             ["one seed", "replicate of a seed"], frameon=False, loc="lower right", fontsize=6)
fig.savefig(os.path.join(OUT, "results_models.pdf")); fig.savefig(os.path.join(OUT, "results_models.png"), dpi=220)

for m in MODELS:
    print(f"{NAME[m]:22} cleanup runs {len(cl[m])} mean {st.mean(sum(v) for _, v in cl[m]) if cl[m] else float('nan'):7.1f} | "
          f"harvest runs {len(hv[m])} norm {st.mean(x for _, x in norm[m]) if norm[m] else float('nan'):.2f}")
print("wrote", OUT + "/results_trials.{pdf,png}", "and", OUT + "/results_models.{pdf,png}")
