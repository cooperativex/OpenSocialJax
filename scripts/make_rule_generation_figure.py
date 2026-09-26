"""How the hidden rules are generated: one flat paper figure per environment (7.0 in wide, print-size fonts), four panels
read left to right:  1 rule space  ->  2 one sampled rule set (seed 906)  ->  3 benchmark and train / held-out split
->  4 what one meta-episode draws.  Both figures share the panel widths, so they stack into one figure.
Facts (rule sets, sizes) come from scripts/rule_generation_data.json (written by rule_generation_data.py, needs JAX);
this script needs no JAX.   usage: python scripts/make_rule_generation_figure.py --env harvest|cleanup"""
import argparse, json, sys
sys.path.insert(0, "scripts")
from figkit import Fig, BLACK, GREY, DIM, LINE, BLUE, ORANGE, HUE_RGB, CARD

ap = argparse.ArgumentParser(); ap.add_argument("--env", choices=["harvest", "cleanup"], default="harvest"); a = ap.parse_args()
DATA = json.load(open("scripts/rule_generation_data.json")); SEED = DATA["seed"]; d = DATA[a.env]
SW, SH = 7.0, 1.98
S, SM, HH = 7.3, 6.4, 0.19                                     # text sizes (pt), header height (in)
W = ([1.84, 1.42, 2.22, 1.26] if a.env == "cleanup" else [1.90, 1.92, 1.22, 1.38]); GAP = 0.14; X = [0.05]
for w in W[:-1]: X.append(X[-1] + w + GAP)
TOP, H = 0.05, SH - 0.10
ICON = "docs/figures/harvest_icons/{}_{}.png"; STAGES = ("unripe", "ripe", "overripe"); STAGE_WORD = ("unripe", "ripe", "over-ripe")
TRAIN_C, HELD_C = (0xC9, 0xD8, 0xEE), (0xF4, 0xC9, 0x9B)
F = Fig(SW, SH)
fmt = lambda n: f"{n:,}"


def split_bar(x, y, w, total, train, held, what):
    """'what' sets -> a bar split train | held-out, with the counts under it."""
    wt = w * train / (train + held)
    F.box(x, y, wt, 0.17, fill=TRAIN_C, line=LINE, radius=0, width=0.5); F.box(x + wt, y, w - wt, 0.17, fill=HELD_C, line=ORANGE, radius=0, width=0.75)
    F.text(x, y, wt, 0.17, [[("train", False)]], size=SM, align="c", anchor="m")
    F.text(x, y + 0.18, w, 0.13, [[(fmt(train), False, GREY)]], size=SM)
    F.text(x, y + 0.18, w, 0.13, [[("held-out  ", True, ORANGE), (fmt(held), True, ORANGE)]], size=SM, align="r")


def trials(x, y, w):
    bw, bh, g = min(0.36, (w - 0.30) / 3), 0.16, 0.09; n = 3; x0 = x + (w - (n * bw + (n - 1) * g)) / 2
    for i, lab in enumerate(("trial 1", "trial 2", "…")):
        F.box(x0 + i * (bw + g), y, bw, bh, fill=(0xFF, 0xFF, 0xFF), line=LINE, radius=0.2, width=0.75)
        F.text(x0 + i * (bw + g), y, bw, bh, [[(lab, False)]], size=SM, align="c", anchor="m")
        if i: F.line(x0 + i * (bw + g) - g, y + bh / 2, x0 + i * (bw + g), y + bh / 2, width=0.9, head_len=0.06)


if a.env == "harvest":
    HUES = d["hues"]; drawn = d["drawn"]
    # ---------------------------------------------------------------- 1 rule space
    y = F.card(X[0], TOP, W[0], H, 1, "Rule space", "per colour")
    lab, cw = 0.66, 0.40; cx = [X[0] + lab + i * cw for i in range(3)]
    F.text(X[0] + 0.03, y + 0.10, lab, 0.26, ["apple", "eaten at"], size=SM, colour=GREY)
    for i in range(3):
        F.picture(ICON.format("yellow", STAGES[i]), cx[i] + (cw - 0.24) / 2, y + 0.05, 0.24, smooth=True)
        F.text(cx[i] - 0.05, y + 0.29, cw + 0.10, 0.12, [[(STAGE_WORD[i], False, GREY)]], size=SM - 0.4, align="c")
    for k, (name, vals) in enumerate((("it pays", "1 · 0.5 · 0.25"), ("its cell regrows", "fast · normal · slow"))):
        yy = y + 0.47 + k * 0.42
        F.text(X[0] + 0.03, yy, lab + 0.05, 0.17, [[(name, True)]], size=SM + 0.3, anchor="m")
        for i in range(3): F.pill(cx[i] + 0.06, yy, cw - 0.12, 0.17, "?", size=S)
        F.text(X[0] + 0.03, yy + 0.19, W[0] - 0.06, 0.13, [[("a hidden order of  ", False, GREY), (vals, True, (0x7A, 0x5A, 0x00))]], size=SM, align="r")
    yy = y + 1.33
    F.text(X[0], yy, W[0], 0.13, [[("6 × 6 orders per colour, 5 colours", False, GREY)]], size=SM, align="c")
    F.text(X[0], yy + 0.14, W[0], 0.16, [[("36", True), ("5", True, None, True), (" = " + fmt(d["space"]) + " rule sets", True)]], size=S + 0.5, align="c")

    # ---------------------------------------------------------------- 2 one rule set
    y = F.card(X[1], TOP, W[1], H, 2, "One rule set", f"seed {SEED}")
    ic, cw, gap = 0.27, 0.25, 0.07; px = X[1] + 0.04 + ic; rx = px + 3 * cw + gap
    F.text(px, y + 0.03, 3 * cw, 0.12, [[("pays", True)]], size=SM, align="c"); F.text(rx, y + 0.03, 3 * cw, 0.12, [[("cell regrows", True)]], size=SM, align="c")
    for i in range(3):
        for x0 in (px, rx): F.picture(ICON.format("yellow", STAGES[i]), x0 + i * cw + (cw - 0.15) / 2, y + 0.16, 0.15, smooth=True)
    for h, hue in enumerate(HUES):
        yy = y + 0.35 + h * 0.19; on = h in drawn; col = BLACK if on else DIM
        F.picture(ICON.format(hue, "ripe"), X[1] + 0.07, yy + 0.01, 0.16, smooth=True)
        for i in range(3):
            p = d["pay"][h][i]; F.text(px + i * cw, yy, cw, 0.18, [[(f"{p:g}", p == 1.0 and on, col)]], size=S - 0.3, align="c", anchor="m")
            F.text(rx + i * cw, yy, cw, 0.18, [[(d["speed"][h][i][0].upper(), d["speed"][h][i] == "fast" and on, col)]], size=S - 0.3, align="c", anchor="m")
    yy = y + 0.35 + 5 * 0.19 + 0.04
    F.text(X[1], yy, W[1], 0.12, [[("F fast ×2 · N normal ×1 · S slow ×0.5", False, GREY)]], size=SM - 0.2, align="c")
    F.text(X[1], yy + 0.13, W[1], 0.12, [[("grey rows: colours not drawn for this map", False, GREY)]], size=SM - 0.2, align="c")

    # ---------------------------------------------------------------- 3 benchmark
    y = F.card(X[2], TOP, W[2], H, 3, "Benchmark")
    F.text(X[2], y + 0.06, W[2], 0.12, [[("sampled once, fixed seed", False, GREY)]], size=SM, align="c")
    F.text(X[2], y + 0.20, W[2], 0.20, [[(fmt(d["bench"]), True)]], size=10.5, align="c")
    F.text(X[2], y + 0.40, W[2], 0.26, [[("rule sets", False)], [("out of the " + fmt(d["space"]), False, GREY)]], size=SM, align="c", spacing=1.05)
    split_bar(X[2] + 0.08, y + 0.74, W[2] - 0.16, d["bench"], d["train"], d["held_out"], "rule sets")
    F.text(X[2], y + 1.20, W[2], 0.40, [[("all reported runs use", False, GREY)], [("held-out rule sets", True, ORANGE)], [("the model never met", False, GREY)]], size=SM, align="c", spacing=1.05)

    # ---------------------------------------------------------------- 4 one meta-episode
    y = F.card(X[3], TOP, W[3], H, 4, "Meta-episode")
    tw = 1.20; th = F.picture("docs/figures/llm_agent_frame.png", X[3] + (W[3] - tw) / 2, y + 0.05, tw, smooth=True)
    yy = y + 0.05 + th + 0.03
    F.text(X[3], yy, W[3], 0.26, [[("draws ", False), ("1 held-out rule set", True, ORANGE)], [("+ 3 of the 5 colours, one per patch", False)]], size=SM, align="c", spacing=1.05)
    trials(X[3], yy + 0.30, W[3])
    F.text(X[3], yy + 0.49, W[3], 0.26, [[("rules and colours stay; the orchard", False, GREY)], [("is re-laid, the agent's memory kept", False, GREY)]], size=SM - 0.2, align="c", spacing=1.05)

if a.env == "cleanup":
    REC = d["recipes"]; order = ["red-light", "yellow-light", "green-light", "cyan-light", "purple-light", "mid", "dark"]
    CI = "docs/figures/cleanup_icons/{}.png"; HUES5 = ("red", "yellow", "green", "cyan", "purple"); SHADES = ("light", "mid", "dark")
    MIDC, DARKC = (0xC4, 0xC4, 0xC4), (0x5A, 0x5A, 0x5A)
    style = {k: (HUE_RGB[k.split("-")[0]], k[0].upper(), BLACK) for k in order[:5]}; style["mid"] = (MIDC, "M", BLACK); style["dark"] = (DARKC, "D", (0xFF, 0xFF, 0xFF))

    def grid(x, y, filled=None, c=0.19, ic=0.14):
        """4 x 4 ordered pairs of pick-ups at the stations: row = first pick-up, column = second pick-up (station icons)."""
        F.text(x, y - 0.31, 4 * c, 0.12, [[("2nd pick-up →", False, GREY)]], size=SM - 0.6, align="c")
        F.text(x - 0.42, y - 0.33, 0.42, 0.24, [[("1st", False, GREY)], [("pick-up ↓", False, GREY)]], size=SM - 1.0, align="r", spacing=1.0)
        for i in range(4):
            F.picture(CI.format(f"station_{i}"), x + i * c + (c - ic) / 2, y - 0.17, ic, smooth=True)
            F.picture(CI.format(f"station_{i}"), x - 0.18, y + i * c + (c - ic) / 2, ic, smooth=True)
            for j in range(4):
                hit = (filled or {}).get((i, j))
                F.box(x + j * c, y + i * c, c, c, fill=hit[0] if hit else (0xFF, 0xFF, 0xFF), line=LINE, radius=0, width=0.5)
                if hit: F.text(x + j * c, y + i * c, c, c, [[(hit[1], True, hit[2])]], size=SM - 0.3, align="c", anchor="m")

    # ---------------------------------------------------------------- 1 the pieces: 15 waste colours, 4 stations, pairs
    y = F.card(X[0], TOP, W[0], H, 1, "Waste, stations, pick-ups")
    ic = 0.14; gx = X[0] + 0.46; cw = 0.20
    F.text(X[0] + 0.03, y + 0.05, 0.44, 0.26, [[("15 waste", True)], [("colours", True)]], size=SM - 0.5, spacing=1.0)
    F.text(X[0] + 0.03, y + 0.30, 0.44, 0.30, [[("hue × shade;", False, GREY)], [("only water", False, GREY)], [("stops polluting", False, GREY)]], size=SM - 1.3, spacing=1.0)
    for j, hue in enumerate(HUES5):
        F.text(gx + j * cw, y + 0.02, cw, 0.11, [[(hue, False, GREY)]], size=SM - 1.3, align="c")
    for i, sh in enumerate(SHADES):
        yy = y + 0.13 + i * 0.16
        for j, hue in enumerate(HUES5):
            F.picture(CI.format(f"waste_{hue}_{sh}"), gx + j * cw + (cw - ic) / 2, yy, ic, smooth=True)
        F.text(gx + 5 * cw, yy, 0.30, ic, [[(sh, False, GREY)]], size=SM - 1.1, anchor="m")
    yy = y + 0.13 + 3 * 0.16 + 0.06
    F.text(X[0] + 0.03, yy, 0.44, 0.26, [[("4 stations", True)], [("a pick-up takes", False, GREY)], [("that base tool", False, GREY)]], size=SM - 1.1, spacing=1.0)
    for i in range(4):
        F.picture(CI.format(f"station_{i}"), gx + i * cw + (cw - ic) / 2, yy, ic, smooth=True)
        F.text(gx + i * cw, yy + ic, cw, 0.10, [[(f"T{i}", False, GREY)]], size=SM - 1.3, align="c")
    F.text(gx + 4 * cw - 0.01, yy - 0.01, W[0] - (gx - X[0]) - 4 * cw, 0.30, [[("2 pick-ups", False, GREY)], [("= a pair,", False, GREY)], [("[2, 3]", True), (" = T2, T3", False, GREY)]], size=SM - 1.3, spacing=1.0)
    yy += 0.32
    F.text(X[0] + 0.03, yy, W[0] - 0.06, 0.40, [[("7 pairs are the working tools: 5 hue tools", False)], [("clear light waste of one hue; MID / DARK turn", False)], [("any mid / dark cell one shade lighter. The", False)], [("other 9 pairs do nothing.", False)]], size=SM - 1.1, spacing=1.0, colour=GREY)

    # ---------------------------------------------------------------- 2 rule space
    y = F.card(X[1], TOP, W[1], H, 2, "Rule space")
    gx, gy = X[1] + 0.44, y + 0.40; grid(gx, gy)
    tx = gx + 4 * 0.19 + 0.04; tw = X[1] + W[1] - tx - 0.01
    F.text(tx, y + 0.30, tw, 0.60, [[("16 pairs", True)], [("7 are the", False, GREY)], [("tools, all", False, GREY)], [("different", False, GREY)]], size=SM - 1.4, spacing=1.0)
    yy = y + 1.26
    F.text(X[1], yy, W[1], 0.13, [[("a rule set = which 7 pairs", False, GREY)]], size=SM - 0.6, align="c")
    F.text(X[1], yy + 0.14, W[1], 0.16, [[("P(16, 7) = " + fmt(d["space"]), True)]], size=S - 0.2, align="c")

    # ---------------------------------------------------------------- 3 one rule set, and what a meta-episode draws
    y = F.card(X[2], TOP, W[2], H, 3, "One rule set", f"seed {SEED}, hidden")
    gx, gy = X[2] + 0.44, y + 0.40; grid(gx, gy, {tuple(REC[k]): style[k] for k in order})
    icon = {k: CI.format("waste_" + k.replace("-", "_")) for k in order[:5]}; icon["mid"] = CI.format("waste_cyan_mid"); icon["dark"] = CI.format("waste_cyan_dark")
    effect = {k: "→ water" for k in order[:5]}; effect["mid"] = "→ light"; effect["dark"] = "→ mid"
    lx = gx + 4 * 0.19 + 0.07; lw = X[2] + W[2] - lx - 0.02
    F.text(lx, y + 0.02, lw, 0.12, [[("pair → the shot's effect", True)]], size=SM - 1.2)
    for i, k in enumerate(order):
        yy = y + 0.16 + i * 0.15
        F.box(lx, yy + 0.02, 0.09, 0.09, fill=style[k][0], line=None, radius=0)
        F.text(lx + 0.09, yy, 0.28, 0.13, [[(f"[{REC[k][0]}, {REC[k][1]}]", True)]], size=SM - 1.0, anchor="m")
        F.picture(icon[k], lx + 0.37, yy + 0.005, 0.12, smooth=True)
        F.text(lx + 0.48, yy, lw - 0.48, 0.13, [[(effect[k], False)]], size=SM - 1.5, anchor="m")
    yy = y + 1.26
    F.text(X[2], yy, W[2], 0.28, [[("MID / DARK act on any hue. The agent never sees this table:", False, GREY)],
                                 [("it crafts a pair, fires at waste and reads off what the shot did", False, GREY)]], size=SM - 1.4, align="c", spacing=1.05)

    # ---------------------------------------------------------------- 4 the map this rule set was drawn with
    y = F.card(X[3], TOP, W[3], H, 4, "Meta-episode")
    mw = W[3] - 0.16; mh = F.picture("docs/figures/llm_agent_frame_cleanup.png", X[3] + 0.08, y + 0.04, mw, smooth=True)
    yy = y + 0.04 + mh + 0.04
    F.text(X[3] + 0.02, yy, W[3] - 0.04, 0.60,
           [[("draws the rule set (3),", False, GREY)], [("a ", False, GREY), ("held-out", True, ORANGE), (" map and a", False, GREY)], [("main hue; all kept for", False, GREY)], [("every trial of the run", False, GREY)]], size=SM - 1.3, spacing=1.0)
    F.text(X[3] + 0.02, yy + 0.44, W[3] - 0.04, 0.30, [[("river", True), (" top left · ", False, GREY), ("orchard", True)], [("right · ", False, GREY), ("stations", True), (" T0–T3", False, GREY)]], size=SM - 1.3, spacing=1.0)

for i in range(len(W) - 1):                                     # panel -> panel
    F.line(X[i] + W[i] + 0.02, TOP + H / 2, X[i + 1] - 0.02, TOP + H / 2, width=1.25, head_len=0.08)
sfx = "" if a.env == "harvest" else "_cleanup"
F.save(f"docs/rule_generation{'_' + a.env}_2026-09-21.pptx", f"docs/figures/rule_generation_{a.env}.png")
