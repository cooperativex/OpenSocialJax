"""How restraint unravels across trials in a Harvest formal experiment: what the agents plan (from their notebooks) next
to what they do (zaps, orchard lifetime), per trial.

Notebook themes: at the first step of trial t >= 2 every agent's prompt shows the lessons it wrote at the end of trial
t-1 and its plans for trial t. Panel (a) plots HAND-CODED counts read from CODING (docs/analysis/
harvest_notes_coding.json): each of the 15 blocks per trial (3 seeds x 5 agents) was read in full and coded with one
rubric for Claude Opus 5.5, gpt-6-sol and Qwen3.8-27B. The keyword patterns in THEMES are printed only as a rough
cross-check: they were written from Claude's wording and undercount the other models badly, so they are not plotted.
Behaviour: zaps = beam shots per trial (team, mean over seeds); orchard lifetime = step at which no apple was left and
none could regrow (the run fast-forwards from there), 300 = alive to the end.

Writes docs/analysis/harvest_notes_timeline.{pdf,png,json}.

usage: python scripts/open_harvest_notes_timeline.py [EXP_DIR]"""
import json, os, re, statistics as st, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = sys.argv[1] if len(sys.argv) > 1 else "experiments/open_harvest_5ag"
OUT = "docs/analysis/harvest_notes_timeline"
CODING = "docs/analysis/harvest_notes_coding.json"
cfg = json.load(open(os.path.join(EXP, "config.json")))
SEEDS, INNER, TRIALS = cfg["setting"]["seeds"], cfg["setting"]["inner"], cfg["setting"]["outer"]

THEMES = {   # name: (section, pattern)
    "plan_guard":   ("plan", r"guard|zap (any|agents?|rivals?|intruders?)[^.]*(enter|arriv|reach|strip|come|my)|protect"),
    "plan_hang":    ("plan", r"keep \d\+?|leave \d|hanging|farm"),
    "plan_measure": ("plan", r"regrow\w*[^.]*(stage|ripe vs|ripe versus|over-ripe vs|over-ripe versus|by the stage)|(stage|ripe vs|ripe versus)[^.]*regrow"),
    "plan_fast":    ("plan", r"eat (every|everything|fast|quickly|at once|immediately|them fast)|as fast as|do not leave any|no need to save|front-load"
                             r"|before (rivals|others|the crowd|A|B|C|D|E)\b"),
    "lesson_futile": ("lesson", r"anyway|regardless|hands? (them |it |the apples )?(to|them)|conserving (just|only)|saving [^.]*(rarely|failed|fail)"
                                r"|restraint only pays|waiting[^.]*(fail|lost|loses|underperform)|conservative|never count on|guarding[^.]*fail"
                                r"|farming[^.]*(rarely|fail)|beats (guarding|farming|waiting)"),
    "lesson_zapcost": ("lesson", r"biggest loss|zaps? (cost|are) (me )?(more|far more|my)|avoid zap|zap duels|zapping (back )?cost"
                                 r"|costs me more than it gains|did not pay|don.t initiate|leaving beats fighting|did not prevent retaliation"),
    "lesson_rot":   ("lesson", r"rot\w*[^.]*regrow"),
}


def run(model, seed):
    d = os.path.join(EXP, "runs", model, f"seed{seed}")
    if not os.path.exists(os.path.join(d, "DONE")):
        return None
    r = json.load(open(os.path.join(d, "result.json")))
    first, zaps, dead = {}, [0] * TRIALS, {}
    for line in open(os.path.join(d, "decisions.jsonl")):
        x = json.loads(line)
        if x.get("event") == "fast_forward":
            dead[int(x["trial"])] = int(x["from_t"]); continue
        if "prompt" in x and (x["agent"], x["trial"]) not in first:
            first[(x["agent"], x["trial"])] = x["prompt"]
        o = x.get("outcome")
        if isinstance(o, dict) and o.get("action") == 6:
            zaps[x["trial"]] += 1
    blocks = {}
    for (i, t), p in first.items():
        if t == 0:
            continue
        a, m, b = p.find("YOUR NOTES"), p.find("What you meant to try"), p.find("EARLIER TRIALS")
        blocks[(i, t)] = {"lesson": p[a:m], "plan": p[m:b]}
    return dict(points=r["team_points_per_trial"], zaps=zaps, alive=[dead.get(t, INNER) for t in range(TRIALS)], blocks=blocks)


data = {}
for model in cfg["models"]:
    R = [x for x in (run(model, s) for s in SEEDS) if x]
    if not R:
        continue
    themes = {k: [None] + [sum(bool(re.search(pat, r["blocks"][key][sec], re.I)) for r in R for key in r["blocks"] if key[1] == t)
                           / max(1, sum(1 for r in R for key in r["blocks"] if key[1] == t)) for t in range(1, TRIALS)]
              for k, (sec, pat) in THEMES.items()}
    data[model] = dict(seeds=len(R), themes=themes,
                       zaps=[st.mean(r["zaps"][t] for r in R) for t in range(TRIALS)],
                       alive=[st.mean(r["alive"][t] for r in R) for t in range(TRIALS)],
                       points=[st.mean(r["points"][t] for r in R) for t in range(TRIALS)])

for m, v in data.items():
    print(f"{m} ({v['seeds']} seeds)")
    for k in THEMES:
        print(f"  {k:15s}", ["-" if x is None else f"{x:.2f}" for x in v["themes"][k]])
    print(f"  {'zaps':15s}", [round(x, 1) for x in v["zaps"]]); print(f"  {'alive':15s}", [round(x) for x in v["alive"]])
    print(f"  {'points':15s}", [round(x, 1) for x in v["points"]])

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(data, open(OUT + ".json", "w"), indent=1)

# ---- figure: (a) hand-coded plans of three models, (b) zaps per trial, (c) orchard lifetime; all models in (b)-(c), three highlighted
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
HI = {"claude-opus-5.5": ("Claude Opus 5.5", "#D96B00"), "gpt-6-sol": ("gpt-6-sol", "#1F5FBF"), "qwen3.8-27b": ("Qwen3.8-27B", "#3FA64B")}
T = list(range(1, TRIALS + 1))
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6))
fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.33, wspace=0.38)
coding = json.load(open(CODING))["models"]
for m, (lab, col) in HI.items():          # hand-coded share of the 15 blocks; T2..T5
    if m not in coding:
        continue
    ax[0].plot(T[1:], [100 * n / 15 for n in coding[m]["plan_guard"]], "-", color=col, marker="o", ms=3, lw=1.4)
    ax[0].plot(T[1:], [100 * n / 15 for n in coding[m]["plan_fast"]], "--", color=col, marker="s", ms=2.6, lw=1.2)
ax[0].set_ylim(0, 100); ax[0].set_ylabel("% of agents (hand-coded)"); ax[0].set_title("(a) Plans in the notebooks", loc="left")
ax[0].legend([plt.Line2D([], [], color="#595959", ls="-", marker="o", ms=3), plt.Line2D([], [], color="#595959", ls="--", marker="s", ms=2.6)],
             ["guard a patch / zap intruders", "eat fast, before rivals"], frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.15),
             ncol=1, handlelength=2.2)
for j, key, title, ylab in [(1, "zaps", "(b) Zaps per trial (team)", "zaps"), (2, "alive", "(c) Orchard lifetime", "step orchard died")]:
    for m, v in data.items():
        if m in HI:
            continue
        ax[j].plot(T, v[key], color="#BBBBBB", lw=0.8, zorder=1)
    for m, (lab, col) in HI.items():
        if m in data:
            ax[j].plot(T, data[m][key], color=col, marker="o", ms=3, lw=1.4, label=lab, zorder=2)
    ax[j].set_title(title, loc="left"); ax[j].set_ylabel(ylab)
ax[1].plot([], [], color="#BBBBBB", lw=0.8, label="other 5 models")
h, l = ax[1].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="upper center", bbox_to_anchor=(0.69, 0.245), ncol=2, handlelength=1.6, columnspacing=1.2)
for a in ax:
    a.set_xticks(T); a.set_xticklabels([f"T{t}" for t in T]); a.set_xlim(0.7, TRIALS + 0.3)
ax[0].set_xlim(1.7, TRIALS + 0.3)
fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".png", dpi=200)
print("wrote", OUT + ".{pdf,png,json}")
