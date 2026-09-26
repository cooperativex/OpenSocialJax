"""Schelling diagrams of the role experiments (experiments/open_{cleanup,harvest}_schelling), one figure per environment.

x = number of OTHER agents who are cooperators (0-4), so the two lines are aligned: at each x the vertical gap is what
an agent gains by defecting rather than cooperating, given the same others. A cooperator in a run with k cooperators
sits at x = k - 1, a defector at x = k. y = an agent's own return per trial (OpenCleanup: apples; OpenHarvest: points),
the mean over the agents of that role in all seeds (means only). Only the two lines and their legend are drawn (no title,
no reference line; the authors's choice). The printout also gives the per-agent return of gpt-6-luna without a role in the
formal experiments (team / 5, same seeds, same trials) for the text.

--trials N uses each run's first N trials (only runs that have finished them); per-trial returns are averaged over them.
Per-agent returns come from the step log (harvest: points of each eat) and, for Clean Up, from the trial summaries every
agent is shown at the next trial (or result.json once a run is DONE).

Writes docs/analysis/schelling_trial{N}_{cleanup,harvest}.{pdf,png}.
usage: python scripts/make_schelling_figures.py --trials 1 [--game cleanup|harvest] [--only T] [--all-done]"""
import argparse, collections, json, os, re, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=1); ap.add_argument("--game", choices=["cleanup", "harvest"])
ap.add_argument("--only", type=int, help="use trial T alone (not the mean of trials 1..T); writes schelling_only_trial{T}_{game}")
ap.add_argument("--all-done", action="store_true", help="each run contributes every trial it has finished so far; writes schelling_alldone_{game}")
A = ap.parse_args()
N = 5 if A.all_done else (A.only or A.trials); OUT = "docs/analysis"; SEEDS = (906, 907, 908); K = range(6)
C_COL, D_COL = "#1F5FBF", "#D96B00"


def per_agent_returns(game, k, s):
    """[(role, [return in trial 1..N])] for one run, or None if it has not finished N trials."""
    d = f"experiments/open_{game}_schelling/runs/k{k}/seed{s}"
    roles = json.load(open(os.path.join(d, "roles.json")))["roles"]
    if os.path.exists(os.path.join(d, "DONE")):
        r = json.load(open(os.path.join(d, "result.json")))
        return [(roles[L], [r["per_agent"][i][t]["return"] for t in range(N)]) for i, L in enumerate("ABCDE")]
    if A.all_done:                                   # a running run: every trial it has finished
        return finished_trials(game, d, roles)
    if game == "harvest":
        pts = collections.defaultdict(lambda: collections.defaultdict(float)); cur = 0
        for line in open(os.path.join(d, "decisions.jsonl")):
            x = json.loads(line); o = x.get("outcome")
            if isinstance(o, dict):
                pts[x["agent"]][x["trial"]] += o.get("points", 0); cur = max(cur, x["trial"])
        if cur < N:
            return None
        return [(roles[L], [pts[i][t] for t in range(N)]) for i, L in enumerate("ABCDE")]
    last = {}
    for line in open(os.path.join(d, "decisions.jsonl")):
        x = json.loads(line)
        if "prompt" in x:
            last[x["agent"]] = x["prompt"]
    out = []
    for i, L in enumerate("ABCDE"):
        got = {int(t): int(a) for t, _, a in re.findall(r"^Trial (\d): you cleared (\d+) waste cells, ate (\d+) apples", last.get(i, ""), re.M)}
        if any(t not in got for t in range(1, N + 1)):
            return None
        out.append((roles[L], [got[t] for t in range(1, N + 1)]))
    return out


def finished_trials(game, d, roles):
    """[(role, [return in each finished trial])] for a run that is still going (its current trial left out)."""
    if game == "harvest":
        pts = collections.defaultdict(lambda: collections.defaultdict(float)); cur = 0
        for line in open(os.path.join(d, "decisions.jsonl")):
            x = json.loads(line); o = x.get("outcome")
            if isinstance(o, dict):
                pts[x["agent"]][x["trial"]] += o.get("points", 0); cur = max(cur, x["trial"])
        return [(roles[L], [pts[i][t] for t in range(cur)]) for i, L in enumerate("ABCDE")]
    last = {}
    for line in open(os.path.join(d, "decisions.jsonl")):
        x = json.loads(line)
        if "prompt" in x:
            last[x["agent"]] = x["prompt"]
    return [(roles[L], [int(a) for _, _, a in re.findall(r"^Trial (\d): you cleared (\d+) waste cells, ate (\d+) apples", last.get(i, ""), re.M)])
            for i, L in enumerate("ABCDE")]


def baseline(game):
    """gpt-6-luna without roles, formal runs: per-agent return per trial over the first N trials (team / 5); trial N alone with --only."""
    exp = "open_cleanup_5ag" if game == "cleanup" else "open_harvest_5ag"
    key = "team_apples_per_trial" if game == "cleanup" else "team_points_per_trial"
    vals = [json.load(open(f"experiments/{exp}/runs/gpt-6-luna/seed{s}/result.json"))[key][N - 1:N] if A.only else
            json.load(open(f"experiments/{exp}/runs/gpt-6-luna/seed{s}/result.json"))[key][:N] for s in SEEDS]
    return st.mean(v / 5 for run in vals for v in run)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
for game, title, ylab in (("cleanup", "OpenCleanup", "apples per agent per trial"), ("harvest", "OpenHarvest", "points per agent per trial")):
    if A.game and game != A.game:
        continue
    coop = collections.defaultdict(list); defe = collections.defaultdict(list); missing = []
    for k in K:
        for s in SEEDS:
            r = per_agent_returns(game, k, s)
            if r is None:
                missing.append((k, s)); continue
            for role, v in r:
                val = v[-1] if A.only else st.mean(v)    # --only T: trial T alone
                if role == "cooperator":
                    coop[k - 1].append(val)              # a cooperator sees k - 1 other cooperators
                else:
                    defe[k].append(val)                  # a defector sees k
    fig, ax = plt.subplots(figsize=(3.4, 2.45)); fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.17)
    for data, col, lab in ((coop, C_COL, "cooperator"), (defe, D_COL, "defector")):
        xs = sorted(data)
        ax.plot(xs, [st.mean(data[x]) for x in xs], color=col, marker="o", ms=3.5, lw=1.4, label=lab, zorder=3)
    ax.set_xticks(range(5)); ax.set_xlabel("number of other agents who cooperate"); ax.set_ylabel(ylab); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, f"schelling_alldone_{game}" if A.all_done else f"schelling_only_trial{N}_{game}" if A.only else f"schelling_trial{N}_{game}")
    fig.savefig(base + ".pdf"); fig.savefig(base + ".png", dpi=220); plt.close(fig)
    print(f"{game}: runs missing {missing}")
    for x in range(5):
        f = lambda v: f"{st.mean(v):6.1f}" if v else "     -"
        print(f"   others cooperating {x}: cooperator {f(coop[x])} (n={len(coop[x]):2d})   defector {f(defe[x])} (n={len(defe[x]):2d})")
    print(f"   baseline (no role) {baseline(game):.1f}; wrote {base}.{{pdf,png}}")
