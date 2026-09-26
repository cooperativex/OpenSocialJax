"""Overview figure of OpenHarvest in the three-block layout of the OpenCleanup overview:
  1 environment components  ->  2 sample a latent configuration (hidden rules, map)  ->  3 instantiate and interact.
Print size (7.0 in wide), native PowerPoint shapes (editable .pptx) plus a PNG/PDF preview from the same geometry.
Facts come from scripts/open_harvest_overview_data.json (written by harvest_overview_data.py, needs JAX); this script
needs no JAX. The grey stage icons of the rule table are docs/figures/harvest_icons/grey_<stage>.png (yellow, desaturated).
usage: python scripts/make_open_harvest_overview.py"""
import json, sys
sys.path.insert(0, "scripts")
from figkit import Fig, BLACK, GREY, DIM, LINE, ORANGE

D = json.load(open("scripts/open_harvest_overview_data.json"))
SEED, HUES, PAY, SPEED = D["seed"], D["hues"], D["pay"], D["speed"]
ICON = "docs/figures/harvest_icons/{}_{}.png"; EMPTY = "docs/figures/harvest_icons/empty_cell.png"
STG = ("unripe", "ripe", "overripe"); STG_WORD = ("unripe", "ripe", "over-ripe")
EX = "cyan"; XH = HUES.index(EX)                                   # the example hue of block 3
present = {p["hue"] for p in D["patches"]}
best = {h: STG_WORD[PAY[HUES.index(h)].index(1.0)] for h in HUES}
SW, SH = 7.0, 2.25
W = [1.62, 1.98, 3.15]; GAP = 0.10; X = [0.05]
for w in W[:-1]: X.append(X[-1] + w + GAP)
TOP, H = 0.05, SH - 0.10
NODE, NODE_LINE, HI = (0xEE, 0xF3, 0xFA), (0x9D, 0xB2, 0xCF), (0xFF, 0xF3, 0xD0)
F = Fig(SW, SH)


def tw(s, pt, bold=False):
    """Width of a text run in inches, as the PNG preview draws it."""
    return F.d.textlength(s, font=F.font(False, bold, pt)) / F.dpi


def width(parts, pt, lh):
    return sum((lh * 0.95 + 0.01) if isinstance(p, tuple) and p[0] == "@" else tw(p if isinstance(p, str) else p[0], pt, not isinstance(p, str) and p[1])
               for p in parts)


def run(x, y, parts, pt, h=0.13, colour=BLACK):
    """One line of text runs and inline icons, laid left to right: parts = str | (str, bold[, colour]) | ('@', icon path)."""
    for p in parts:
        if isinstance(p, tuple) and p[0] == "@":
            s = h * 0.95; F.picture(p[1], x, y + (h - s) / 2, s, smooth=True); x += s + 0.01; continue
        t, b, c = (p, False, colour) if isinstance(p, str) else (p[0], p[1], p[2] if len(p) > 2 else colour)
        w = tw(t, pt, b)
        F.text(x - 0.03, y, w + 0.08, h, [[(t, b, c)]], size=pt, anchor="m", wrap=False); x += w
    return x


NPT, NLH = 5.0, 0.11


def node(x, y, w, h, lines):
    """A rule node: rounded box, each line a list of runs (see run), centred."""
    F.box(x, y, w, h, fill=NODE, line=NODE_LINE, radius=0.18, width=0.75)
    y0 = y + (h - NLH * len(lines)) / 2
    for i, parts in enumerate(lines):
        run(x + (w - width(parts, NPT, NLH)) / 2, y0 + i * NLH, parts, NPT, h=NLH)


# ================================================================ 1 environment components
x0, y = X[0], F.card(X[0], TOP, W[0], H, 1, "Environment components")
run(x0 + 0.04, y + 0.03, [("Apples: 5 hues × 3 stages = 15 types", True)], 6.2)
gx, cw, ic = x0 + 0.06, 0.228, 0.14
for j, hue in enumerate(HUES):
    F.text(gx + j * cw - 0.03, y + 0.18, cw + 0.06, 0.10, [[(hue.capitalize(), False, GREY)]], size=5.0, align="c")
for i in range(3):
    yy = y + 0.28 + i * 0.15
    for j, hue in enumerate(HUES):
        F.picture(ICON.format(hue, STG[i]), gx + j * cw + (cw - ic) / 2, yy, ic, smooth=True)
    F.text(gx + 5 * cw, yy, 0.42, ic, [[(STG_WORD[i], False, GREY)]], size=5.4, anchor="m")
yy = y + 0.28 + 3 * 0.15 + 0.005
F.text(x0 + 0.04, yy, W[0] - 0.08, 0.11, [[("each stage lasts 15–20 steps; over-ripe apples rot", False, GREY)]], size=5.0)
yy += 0.13
F.box(x0 + 0.06, yy + 0.02, 0.15, 0.15, fill=None, line=LINE, radius=0, width=0.5); F.picture(EMPTY, x0 + 0.06, yy + 0.02, 0.15, smooth=True)
F.text(x0 + 0.24, yy, W[0] - 0.27, 0.34, [[("Empty tree cell", True), (": grows a new", False)], [("unripe apple, but only while apples", False)],
                                         [("still hang within 2 cells", False)]], size=5.2, spacing=0.95)
yy += 0.345
# the zap beam: the agent faces down (its sprite is front-on); the beam covers the faced cell, the one beyond it and the
# two cells beside the faced cell -- 4 cells, as in the env (one_step, two_step, forward-left, forward-right targets)
zc = 0.08; zx, zy = x0 + 0.035, yy - 0.005
for r in range(3):
    for c in range(3):
        F.box(zx + c * zc, zy + r * zc, zc, zc, fill=(0xE8, 0xE2, 0xD0), line=(0xFF, 0xFF, 0xFF), radius=0, width=0.5)
F.picture("docs/figures/harvest_icons/agent_a.png", zx + zc, zy, zc, smooth=True)
for r, c in ((1, 0), (1, 1), (1, 2), (2, 1)):
    F.picture("docs/figures/harvest_icons/beam.png", zx + c * zc, zy + r * zc, zc, smooth=True)
AG = [[("Agents", True), (" eat by stepping on", False)], [("apples; a ", False), ("zap", True), (" hits the 4 cells", False)],
      [("shown and sends a hit agent", False)], [("to spawn for 12 steps", False)]]
for line in AG:
    assert width(line, 5.2, 0.11) < W[0] - 0.33, line
F.text(x0 + 0.29, yy - 0.02, W[0] - 0.32, 0.44, AG, size=5.2, spacing=0.95)
yy += 0.43
F.text(x0 + 0.04, yy, W[0] - 0.08, 0.34, [[("Hidden, per hue and stage:", True)], [("points ", False), ("1 · 0.5 · 0.25", True, ORANGE)],
                                         [("regrowth speed ", False), ("×2 · ×1 · ×0.5", True, ORANGE)]], size=5.2, spacing=1.0)

# ================================================================ 2 sample a latent configuration
x0, y = X[1], F.card(X[1], TOP, W[1], H, 2, "Sample a latent configuration")
run(x0 + 0.04, y + 0.02, [("(a) Hidden rule sampling", True)], 6.2)
hx = x0 + 0.10; cw = 0.20; px = x0 + 0.32; rx = px + 3 * cw + 0.12; ry = y + 0.19
F.text(px - 0.05, ry, 3 * cw + 0.10, 0.11, [[("points when eaten", True)]], size=5.1, align="c")
F.text(rx - 0.05, ry, 3 * cw + 0.10, 0.11, [[("cell regrowth speed", True)]], size=5.1, align="c")
for i in range(3):
    for x1 in (px, rx):
        F.picture(f"docs/figures/harvest_icons/grey_{STG[i]}.png", x1 + i * cw + (cw - 0.11) / 2, ry + 0.12, 0.11, smooth=True)
        F.text(x1 + i * cw - 0.04, ry + 0.23, cw + 0.08, 0.08, [[(STG_WORD[i], False, GREY)]], size=4.0, align="c")
pitch = 0.12; r0 = ry + 0.31
for h, hue in enumerate(HUES):
    yy = r0 + h * pitch; on = hue in present; col = BLACK if on else DIM
    if hue == EX:
        F.box(x0 + 0.06, yy - 0.005, W[1] - 0.12, pitch, fill=None, line=ORANGE, radius=0.25, width=0.9)
    F.picture(ICON.format(hue, "ripe"), hx, yy + 0.012, 0.105, smooth=True)
    for i in range(3):
        p = PAY[h][i]; s = SPEED[h][i]
        if on and p == 1.0: F.box(px + i * cw + 0.03, yy + 0.014, cw - 0.06, pitch - 0.028, fill=HI, line=None, radius=0.3)
        F.text(px + i * cw, yy, cw, pitch, [[(f"{p:g}", p == 1.0 and on, col)]], size=5.6, align="c", anchor="m")
        F.text(rx + i * cw, yy, cw, pitch, [[(f"×{s:g}", s == 2.0 and on, col)]], size=5.6, align="c", anchor="m")
yy = r0 + 5 * pitch + 0.01
F.text(x0, yy, W[1], 0.10, [[(f"seed {SEED} shown; grey row: {', '.join(h for h in HUES if h not in present)} is on no patch of its map", False, GREY)]], size=4.6, align="c")
F.text(x0, yy + 0.10, W[1], 0.11, [[("each hue: 3! point orders × 3! speed orders", False, GREY)]], size=5.0, align="c")
F.text(x0, yy + 0.21, W[1], 0.13, [[("36", True), ("5", True, None, True), (" = 60,466,176 rule sets", True)]], size=6.0, align="c")
yy += 0.38
run(x0 + 0.04, yy, [("(b) Map sampling", True)], 6.2)
yy += 0.15
xx = run(x0 + 0.06, yy, [("Patch hues, with replacement:", False)], 5.3, h=0.15)
for k, p in enumerate(D["patches"]):
    F.box(xx + 0.04 + k * 0.155, yy, 0.145, 0.15, fill=(0xFF, 0xFF, 0xFF), line=LINE, radius=0.2, width=0.5)
    F.picture(ICON.format(p["hue"], "ripe"), xx + 0.04 + k * 0.155 + 0.015, yy + 0.017, 0.115, smooth=True)
yy += 0.18
F.text(x0 + 0.04, yy, W[1] - 0.08, 0.12, [[("Layout", True), (": patch positions and turns, spawn row", False)]], size=5.2)

# ================================================================ 3 instantiate and interact
# Left ~40 %: the example map and the neighbour rule under it. Right ~60 %: the life of one tree cell, which branches
# after RIPEN into HARVEST -> REGROW AFTER HARVEST and ROT; both branches bring a new unripe apple back to RIPEN.
x0, y = X[2], F.card(X[2], TOP, W[2], H, 3, "Instantiate and interact")
BPT, BLH = 4.7, 0.10                                                              # node text size / line height
BOX = {
    "ripen":   [[("RIPEN", True)], ["unripe → ripe → overripe"], [("15–20 steps per stage", False, GREY)]],
    "harvest": [[("HARVEST", True)], ["reward and regrowth multiplier"], ["depend on colour and stage"]],
    "rot":     [[("ROT", True)], ["regrow in 2–5 steps,"], ["independent of nearby apples"]],
    "regrow":  [[("REGROW AFTER HARVEST", True)], ["p = base(n) × multiplier"]],
}


def bnode(x, yy, w, h, lines):
    F.box(x, yy, w, h, fill=NODE, line=NODE_LINE, radius=0.18, width=0.75)
    y0 = yy + (h - BLH * len(lines)) / 2
    for k, parts in enumerate(lines):
        run(x + (w - width(parts, BPT, BLH)) / 2, y0 + k * BLH, parts, BPT, h=BLH)


need = lambda *keys: max(width(l, BPT, BLH) for k in keys for l in BOX[k]) + 0.08
c1, c2, cg = need("harvest", "regrow"), need("rot"), 0.05
xr = x0 + W[2] - 0.05                                                            # right edge of the diagram
R0 = xr - (c1 + cg + c2); lane = 0.10                                            # the return lane runs in the gap left of R0
mw = R0 - lane - (x0 + 0.05); mh = mw * D["grid"][0] / D["grid"][1]
run(x0 + 0.04, y + 0.02, [("Example configuration", True)], 6.2)
F.picture(D["map_png"], x0 + 0.05, y + 0.17, mw, smooth=False)
# base(n), under the map
by0 = y + 0.17 + mh + 0.06; bh_ = 0.47
F.box(x0 + 0.05, by0, mw, bh_, fill=(0xFF, 0xFF, 0xFF), line=LINE, radius=0.10, width=0.5, dash=True)
F.text(x0 + 0.05, by0 + 0.03, mw, 0.21, [[("base(n)", True), (", where n is the number of", False)], [("apples within two cells", False)]], size=BPT + 0.2, align="c", spacing=1.0)
for k, t in enumerate(("n ≥ 3: 2.5% per step", "n = 2: 0.5% per step", "n = 1: 0.1% per step", "n = 0: no regrowth")):
    F.text(x0 + 0.09 + (k % 2) * (mw - 0.08) / 2, by0 + 0.24 + (k // 2) * 0.10, (mw - 0.08) / 2, 0.10, [[(t, False)]], size=BPT - 0.2, align="c")
# the branching life of a tree cell
rt, rh = y + 0.03, 0.34                                                          # RIPEN
h1 = rt + rh + 0.19; hh = 0.34                                                    # HARVEST and ROT
g1 = h1 + hh + 0.17; gh = 0.26                                                    # REGROW AFTER HARVEST
merge = g1 + gh + 0.13
bnode(R0, rt, xr - R0, rh, BOX["ripen"])
bnode(R0, h1, c1, hh, BOX["harvest"]); bnode(R0 + c1 + cg, h1, c2, hh, BOX["rot"])
bnode(R0, g1, c1, gh, BOX["regrow"])
ax1, ax2 = R0 + 0.14, R0 + c1 + cg + 0.14                                        # arrow x in each column
F.line(ax1, rt + rh + 0.005, ax1, h1 - 0.005, width=0.9, head_len=0.045)
F.text(ax1 + 0.03, rt + rh + 0.035, c1 - 0.2, 0.10, [[("harvest at any stage", False, GREY)]], size=4.5)
F.line(ax2, rt + rh + 0.005, ax2, h1 - 0.005, width=0.9, head_len=0.045)
F.text(ax2 + 0.03, rt + rh + 0.035, c2 - 0.2, 0.10, [[("overripe, not harvested", False, GREY)]], size=4.5)
F.line(ax1, h1 + hh + 0.005, ax1, g1 - 0.005, width=0.9, head_len=0.045)
F.text(ax1 + 0.03, h1 + hh + 0.03, c1 - 0.2, 0.10, [[("harvested tree cell", False, GREY)]], size=4.5)
# both branches bring a new unripe apple back to RIPEN
mx1, mx2 = R0 + c1 / 2, R0 + c1 + cg + c2 / 2; lx = R0 - lane / 2
F.line(mx1, g1 + gh + 0.005, mx1, merge, width=0.9, head=False)
F.line(mx2, h1 + hh + 0.005, mx2, merge, width=0.9, head=False)
F.line(mx2, merge, lx, merge, width=0.9, head=False)
F.line(lx, merge, lx, rt + rh / 2, width=0.9, head=False)
F.line(lx, rt + rh / 2, R0 - 0.005, rt + rh / 2, width=0.9, head_len=0.045)
F.text(mx1 + 0.04, merge - 0.11, mx2 - mx1 - 0.08, 0.10, [[("new unripe apple", False, GREY)]], size=4.5, align="c")
# the one-line challenge
cy = y + H - 0.19 - 0.17
F.text(x0 + 0.04, cy, W[2] - 0.08, 0.13, [[("Agent challenge: ", True), ("infer hidden rules, preserve local apple density, and adapt to other agents.", False)]], size=5.6)

for i in range(len(W) - 1):                                                    # block -> block
    F.line(X[i] + W[i] + 0.015, TOP + H / 2, X[i + 1] - 0.015, TOP + H / 2, width=1.25, head_len=0.07)
F.save("docs/harvest_overview_2026-09-24.pptx", "docs/figures/harvest_overview.png")
