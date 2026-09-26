"""open_harvest harness: one LLM call per agent per step, same prompt skeleton as the cleanup harness (open_cleanup_policy.py).

Blocks: progress / RULES YOU KNOW (what you ate and what it paid, regrowth and rot you watched) / EARLIER TRIALS /
OTHER AGENTS / YOUR PLAN / LAST STEP + events / RECENT / NOW (values, patches in view, map) / ACTIONS NOW.
Everything the model reads is built from its own 11x11 view; the hidden rules (per colour: which stage pays 1 / 0.5 / 0.25, and
which stage an apple must be eaten at for its cell to regrow fast / normal / slow) are never shown unless --reveal-rules (ablation).
"""
from __future__ import annotations

import argparse, json, os, re, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import jax
import numpy as onp

from opensocialjax.environments.open_harvest.open_harvest import (
    Items, NUM_RIPE_STAGES, is_apple, apple_hue, apple_stage, HUE_NAMES)
from opensocialjax.environments.discovery_rules import decode_harvest_ruleset, REGROW_RATES, PAY_ORDERS, STAGE_PAY
from llm_policy import open_harvest_prompt as P
from llm_policy.open_cleanup_policy import ChatProvider, PlanLedger, LETTERS, STALE_AFTER, beam_cells, view_window, parse_reply as _parse_reply_base  # noqa: F401
from llm_policy.open_cleanup_policy import NOTE_SCHEMA, Notebook, parse_note      # the model's own notes: one implementation for both games
from opensocialjax.environments.held_out import held_out_harvest_env

# raw (LLM) actions -> env actions.  env: 0 turn L, 1 turn R, 2 col+1 (east), 3 col-1 (west), 4 row+1 (south), 5 row-1 (north), 6 stay, 7 zap
RAW_TO_ENV = {0: 5, 1: 4, 2: 2, 3: 3, 4: 0, 5: 1, 6: 7, 7: 6}
ZAP, STAY = 6, 7
MOVE_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}
FACING = {0: "south", 1: "east", 2: "north", 3: "west"}
MODES = ["harvest", "wait", "explore", "travel", "zap"]      # zap added 2026-09-19: firing used to be filed under wait/harvest
GOAL_TYPES = ["move_to", "eat", "wait", "zap"]
HUE_LETTER = "RYGCP"
STAGE_LETTER = "gro"
STAGE_WORD = ("unripe", "ripe", "over-ripe")
ROT = 3                      # ledger key for a cell whose apple rotted (regrows by itself), kept apart from cells eaten at over-ripe
LEFT_WORD = STAGE_WORD + ("rotted",)
RATE_WORD = dict(zip((float(r) for r in REGROW_RATES), ("fast", "normal", "slow")))

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        # "why_this_move", not "reasoning", as in open_cleanup_policy: Claude refuses some prompts that ask for a field called
        # "reasoning" (reasoning_extraction). The parsed reply still calls it "reasoning" for the analysis tools.
        "why_this_move": {"type": "string", "maxLength": 500},
        "mode": {"type": "string", "enum": MODES},
        "goal": {"type": "object",
                 "properties": {"type": {"type": "string", "enum": GOAL_TYPES},
                                "cell": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                                "why": {"type": "string", "maxLength": 80}},
                 "required": ["type", "cell", "why"], "additionalProperties": False},
        "note": NOTE_SCHEMA,
        "action": {"type": "integer", "minimum": 0, "maximum": 7},
    },
    "required": ["why_this_move", "mode", "goal", "note", "action"],
    "additionalProperties": False,
}


# small map for the LLM runs (9 x 15): a full vanilla diamond in the middle, quarter-diamonds cut by the walls left and right
MINI_MAP = [
    "RRR         GGG",
    "RR     Y     GG",
    "R     YYY     G",
    "     YYYYY     ",
    "      YYY      ",
    "       Y       ",
    "               ",
    "  QQQQQQQQQQQ  ",
    " PPPPPPPPPPPPP ",
]


def make_env(agents=3, inner=500, outer=10, ripen=25, on_train=False, layout=0, respawn_wait=12,
             ripen_range=(15, 20), rot_range=(2, 5), rates=None):
    """The game the harness plays (the baseline since 2026-09-24): the original Commons Harvest map, 16 x 22 with six
    patches (four diamonds, two quarter diamonds, 64 trees), each patch's colour drawn at random with replacement,
    map number `layout` of the held-out (with on_train, the training) pool in layouts_orig6.py; pick_layout(seed) is
    the map a seed plays on. Individual rewards (the only kind)."""
    from opensocialjax.environments.open_harvest.layouts_orig6 import layout_split
    return _harvest_env(layout_split(on_train).map_ascii(layout), agents, inner, outer, ripen, on_train, respawn_wait,
                        ripen_range, rot_range, rates, patch_hues_with_replacement=True)


POOL = "orig6"                               # the map pool make_env draws from; recorded with every run's results


def _harvest_env(amap, agents, inner, outer, ripen, on_train, respawn_wait, ripen_range, rot_range, rates, **extra):
    return held_out_harvest_env(num_agents=agents, num_inner_steps=inner, num_outer_steps=outer, ripen_steps=ripen, on_train=on_train,
                                map_ASCII=amap, grid_size=(len(amap), max(len(r) for r in amap)),
                                frames_till_respawn=respawn_wait, ripen_range=tuple(ripen_range), rot_regrow_range=tuple(rot_range),
                                regrow_rates=None if rates is None else tuple(rates), **extra)


def tree_mask(env):
    m = onp.zeros((env.GRID_SIZE_ROW, env.GRID_SIZE_COL), dtype=bool)
    sp = onp.array(env.SPAWNS_APPLE); m[sp[:, 0], sp[:, 1]] = True
    return m


# ---------------------------------------------------------------- observation
def present_hues(state):
    """The hues on the map this meta-episode (drawn per patch at reset, fixed across its trials)."""
    return sorted(set(int(h) for h in onp.array(state.cell_hues)))


def hue_grid(env, state):
    """(rows, cols) hue of every tree cell this meta-episode."""
    g = onp.zeros((env.GRID_SIZE_ROW, env.GRID_SIZE_COL), dtype=int)
    sp = onp.array(env.SPAWNS_APPLE); g[sp[:, 0], sp[:, 1]] = onp.array(state.cell_hues)
    return g


def initial_state(env, seed, episode=0):
    """The reset state run_episode will start from (same key split), to read the drawn hues before building prompts."""
    _, kr, _ = jax.random.split(jax.random.PRNGKey(seed + episode), 3)
    return env.reset(kr)[1]


def observe(env, state, i, trees, hue_of_cell):
    """Text of the NOW block and the facts the ledgers and ACTIONS NOW need."""
    grid = onp.array(state.grid); H, W = grid.shape
    locs = onp.array(state.agent_locs); values = onp.array(state.values)[i]
    r0, c0, d = (int(x) for x in locs[i])
    pos_of = {(int(locs[j][0]), int(locs[j][1])): j for j in range(len(locs)) if j != i}
    apples, empties, others, view, rows = {}, {}, {}, set(), []
    r_lo, r_hi, c_lo, c_hi = view_window(env, (r0, c0, d))   # the env's own window: 9 ahead, 1 behind, 5 each side
    for r in range(r_lo, r_hi + 1):
        cells = []
        for c in range(c_lo, c_hi + 1):
            if not (0 <= r < H and 0 <= c < W):
                cells.append("##"); continue
            view.add((r, c)); code = int(grid[r, c])
            if (r, c) == (r0, c0):
                cells.append("ME")
            elif (r, c) in pos_of:
                j = pos_of[(r, c)]; others[j] = (r, c, int(locs[j][2])); cells.append("@" + LETTERS[j])
            elif code == int(Items.wall):
                cells.append("##")
            elif is_apple(code):
                h, s = int(apple_hue(code)), int(apple_stage(code)); apples[(r, c)] = (h, s); cells.append(HUE_LETTER[h] + STAGE_LETTER[s])
            elif trees[r, c]:
                h = int(hue_of_cell[r, c]); empties[(r, c)] = h; cells.append(HUE_LETTER[h].lower() + "_")
            elif code >= len(Items):
                cells.append("@" + LETTERS[code - len(Items)] if code - len(Items) < len(LETTERS) else "@?")
            else:
                cells.append("..")
        rows.append(f"r{r:>2} " + " ".join(cells))
    lines = [f"You are agent {LETTERS[i]}, at ({r0}, {c0}), facing {FACING[d]}."]
    patch = {}
    for (r, c), (h, s) in apples.items():
        patch.setdefault(h, [0, 0, 0, 0]); patch[h][s] += 1
    for (r, c), h in empties.items():
        patch.setdefault(h, [0, 0, 0, 0]); patch[h][3] += 1
    if patch:
        hue_of = {**{rc: h for rc, (h, _) in apples.items()}, **empties}
        parts = []
        for g in patches_of(hue_of):                        # by position: two patches in view can share a colour
            v = [sum(1 for rc in g if rc in apples and apples[rc][1] == st) for st in range(3)] + [sum(1 for rc in g if rc in empties)]
            parts.append(f"{patch_name(g, hue_of)}: {sum(v[:3])} apples (unripe {v[0]}, ripe {v[1]}, over-ripe {v[2]}), {v[3]} empty tree cells")
        lines.append("Patches in view: " + "; ".join(parts) + ".")
    else:
        lines.append("No trees in view.")
    lines.append("Other agents in view: " + (", ".join(f"{LETTERS[j]} at ({r}, {c})" for j, (r, c, _) in sorted(others.items())) if others else "none") + ".")
    lines.append("Map:")
    lines.append("    " + " ".join(f"{c:>2}" if 0 <= c < W else "  " for c in range(c_lo, c_hi + 1)))
    lines.extend(rows)
    beam = [(rc, int(grid[rc])) for rc in beam_cells(env, (r0, c0, d))]
    facts = {"pos": (r0, c0), "facing": d, "values": [int(v) for v in values], "apples": apples, "empties": empties,
             "others": others, "view": view, "beam": beam, "patch": patch, "grid": grid, "pos_of": pos_of}
    return "\n".join(lines), facts


def actions_now(facts, env, zap_cd=0):
    grid = facts["grid"]; r0, c0 = facts["pos"]; items = []
    names = ["north", "south", "east", "west"]
    for a in range(4):
        dr, dc = MOVE_DELTA[a]; r, c = r0 + dr, c0 + dc
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) or int(grid[r, c]) == int(Items.wall):
            what = "wall, blocked"
        elif (r, c) in facts["pos_of"]:
            what = f"agent {LETTERS[facts['pos_of'][(r, c)]]}, blocked"
        elif (r, c) in facts["apples"]:
            h, s = facts["apples"][(r, c)]; what = f"EAT the {HUE_NAMES[h]} {STAGE_WORD[s]} apple there"
        elif (r, c) in facts["empties"]:
            what = f"empty {HUE_NAMES[facts['empties'][(r, c)]]} tree cell"
        else:
            what = "ground"
        items.append(f"{a} {names[a]} -> ({r}, {c}): {what}")
    items += ["4 turn left", "5 turn right"]
    hit = [LETTERS[facts["pos_of"][rc]] for rc, code in facts["beam"] if rc in facts["pos_of"]]
    if zap_cd > 0:
        items.append(f"6 zap (NOT AVAILABLE: the beam is still cooling down, {zap_cd} more step{'s' if zap_cd > 1 else ''})")
    else:
        items.append("6 zap (" + ("beam would hit agent " + ", ".join(hit) if hit else "no agent in the beam") + ")")
    items.append("7 stay")
    return " · ".join(items)


# ---------------------------------------------------------------- ledgers
from opensocialjax.environments.open_harvest.layouts_orig6 import patches_of

NB12 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]   # the env's regrowth neighbourhood


def apples_around(rc, apples):
    """Apples hanging (in view) in the 12-cell neighbourhood the env counts for regrowth of cell rc."""
    return sum(1 for dr, dc in NB12 if (rc[0] + dr, rc[1] + dc) in apples)


def patch_name(cells, hue_of):
    """'yellow rows 6-9 cols 9-11': a patch named by its colour and where its (seen) cells are."""
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    return f"{HUE_NAMES[hue_of[cells[0]]]} rows {min(rs)}-{max(rs)} cols {min(cs)}-{max(cs)}"


def apples_around_if_seen(rc, facts):
    """apples_around, but only when the whole neighbourhood (the part inside the map) is in view; None otherwise, so a
    neighbourhood half out of view is never read as having fewer apples than it has."""
    H, W = facts["grid"].shape
    nb = [(rc[0] + dr, rc[1] + dc) for dr, dc in NB12]
    if any(0 <= r < H and 0 <= c < W and (r, c) not in facts["view"] for r, c in nb):
        return None
    return sum(1 for n in nb if n in facts["apples"])


def _around_change(pairs):
    """'apples around: 4-5 when emptied, 0 at your last look' from [(when emptied, at the last look)]."""
    def rng(xs):
        return f"{min(xs)}" if min(xs) == max(xs) else f"{min(xs)}-{max(xs)}"
    return f"apples around: {rng([a for a, _ in pairs])} when emptied, {rng([b for _, b in pairs])} at your last look"


class HarvestLedger:
    """What this agent has learned itself, kept for the whole episode (across trials): the points each (colour, stage)
    paid it; for every cell it saw being emptied, how long that cell took to regrow; how long apples took to ripen and
    rot when it watched them from the moment they appeared. Facts only, nothing inferred, nothing given."""

    def __init__(self, n_hues=5, hues=None):
        self.n = n_hues; self.hues = list(range(n_hues)) if hues is None else list(hues)
        self.eaten = {}            # (hue, stage) -> (points, count)
        self.regrow = {}           # (hue, stage the cell was emptied at, or ROT) -> {"times": [(steps, exact, around)], "open": [(steps watched, around)]}
        self.ripen = {s: [] for s in range(NUM_RIPE_STAGES)}   # stage -> steps it lasted before the next stage (over-ripe: before rotting); exact only
        self.rot = {h: 0 for h in range(n_hues)}
        self.given = None          # (pay, speed) under --reveal-rules: pay[h][s] points / speed[h][s] regrowth word of colour h at stage s

    def record_eat(self, hue, stage, points, value):
        # facts only: the points this (colour, stage) paid, and how often; no inference about peaks
        pts, c = self.eaten.get((hue, stage), (points, 0))
        self.eaten[(hue, stage)] = (points, c + 1)

    def _slot(self, hue, left):
        return self.regrow.setdefault((hue, left), {"times": [], "open": []})

    def record_regrow(self, hue, left, steps, exact, around):
        """around = (fewest, most) apples seen hanging around the cell while it waited, emptying included."""
        self._slot(hue, left)["times"].append((steps, exact, around))

    def record_open(self, hue, left, steps, around):
        """around = (apples around when it was emptied, apples around at the last look)."""
        self._slot(hue, left)["open"].append((steps, around))

    def record_ripen(self, stage, steps):
        self.ripen[stage].append(steps)

    @staticmethod
    def _what(h, left):
        return f"{HUE_NAMES[h]} cells whose apple rotted" if left == ROT else f"{HUE_NAMES[h]} cells eaten at {STAGE_WORD[left]}"

    def render(self, live=()):
        """live: [(hue, left, steps watched so far, (around when emptied, around at the last look))] for cells still empty
        right now (this trial). The env draws regrowth every step from the apples hanging around the cell AT THAT STEP, so
        the ledger shows how that count moved while the cell waited, not only what it was when the cell was emptied."""
        lines = []
        if self.given is not None:
            pay, speed = self.given
            if pay is None:      # --reveal-regrowth: the regrowth rule only, the points stay hidden
                lines.append("The regrowth rule (given to you, all true): " + "; ".join(
                    f"{HUE_NAMES[h]}: a cell whose apple was eaten " + ", ".join(f"{STAGE_WORD[s]} regrows {speed[h][s]}" for s in range(NUM_RIPE_STAGES)) for h in self.hues)
                    + ". A cell only regrows at all while apples still hang around it.")
            else:
                lines.append("The hidden rules (given to you, all true): " + "; ".join(
                    f"{HUE_NAMES[h]}: " + ", ".join(f"eaten {STAGE_WORD[s]} pays {pay[h][s]:g} and its cell regrows {speed[h][s]}" for s in range(NUM_RIPE_STAGES)) for h in self.hues) + ".")
        ate = []
        for h in self.hues:
            parts = []
            for s in range(NUM_RIPE_STAGES):
                m = self.eaten.get((h, s))
                if m:
                    parts.append(f"{STAGE_WORD[s]} -> {m[0]:g} points (x{m[1]})")
            if parts:
                ate.append(f"{HUE_NAMES[h]}: " + ", ".join(parts))
        lines.append("What you ate and what it paid you: " + ("; ".join(ate) + "." if ate else "nothing yet."))
        pending = {}
        for h, left, steps, around in live:
            pending.setdefault((h, left), []).append((steps, around))
        reg = []
        for key in sorted(set(self.regrow) | set(pending)):
            h, left = key; g = self.regrow.get(key, {"times": [], "open": []}); parts = []
            if g["times"]:
                shown = g["times"][-8:]
                ts = ", ".join((f"{st}" if ex else f"<={st}") for st, ex, _ in shown)
                lo = min(a[0] for _, _, a in g["times"]); hi = max(a[1] for _, _, a in g["times"])
                parts.append(f"{len(g['times'])} regrew after {ts} steps" + (f" (and {len(g['times']) - len(shown)} more)" if len(g["times"]) > len(shown) else "") + f" (apples hanging around them while they waited: {lo}-{hi})")
            if g["open"]:
                parts.append(f"{len(g['open'])} were still empty when their trial ended (up to {max(st for st, _ in g['open'])} steps after being emptied; "
                             + _around_change([a for _, a in g["open"]]) + ")")
            if key in pending:
                parts.append(f"{len(pending[key])} not regrown yet this trial (still empty up to {max(st for st, _ in pending[key])} steps after being emptied; "
                             + _around_change([a for _, a in pending[key]]) + ")")
            reg.append(self._what(h, left) + ": " + "; ".join(parts))
        lines.append("Regrowth you watched (cells you saw emptied; steps until a new apple appeared, <= when you looked away; a cell only grows one while apples hang around it): " + ("\n".join(reg) if reg else "none yet."))
        names = ("unripe -> ripe", "ripe -> over-ripe", "over-ripe -> rotted")
        rip = []
        for st in range(NUM_RIPE_STAGES):
            d = self.ripen[st]
            if d:
                shown = d[-6:]
                rip.append(f"{names[st]} after {', '.join(str(x) for x in shown)} steps" + (f" (x{len(d)})" if len(d) > len(shown) else ""))
        lines.append("Ripening you watched (apples you kept in view from the moment they appeared): " + ("; ".join(rip) + "." if rip else "none yet."))
        rot = [f"{HUE_NAMES[h]} {n}" for h, n in self.rot.items() if n]
        lines.append("Apples you saw rot: " + (", ".join(rot) + "." if rot else "none."))
        return "\n".join(lines)


class SocialLedger:
    """Other agents, from this agent's own view only, this trial: where each was last seen, and what it ate."""

    def __init__(self, me, n):
        self.me, self.n = me, n; self.reset()

    def reset(self):
        self.seen = {}; self.ate = {}; self.hit_me = {}      # ate[j] -> {(hue, stage): count}; hit_me[j] -> times j zapped me

    def update(self, step, others, zapped_js, eats, zapped_me=()):
        for j, (r, c, f) in others.items():
            e = self.seen.setdefault(j, {"count": 0, "zaps": 0})
            e.update(step=step, pos=(r, c), facing=f); e["count"] += 1; e["zaps"] += int(j in zapped_js)
        for j, h, st in eats:
            d = self.ate.setdefault(j, {}); d[(h, st)] = d.get((h, st), 0) + 1
        for j in zapped_me:                                   # who sent me back to a spawn point
            self.hit_me[j] = self.hit_me.get(j, 0) + 1

    def _ate_text(self, j):
        d = self.ate.get(j)
        if not d:
            return "ate nothing in your view"
        return f"ate {sum(d.values())} in your view (" + ", ".join(f"{HUE_NAMES[h]} {STAGE_WORD[st]} {n}" for (h, st), n in sorted(d.items())) + ")"

    def render(self, step):
        lines = []
        for j in range(self.n):
            if j == self.me:
                continue
            e = self.seen.get(j)
            lines.append(f"Agent {LETTERS[j]}: not seen yet this trial." if not e else
                         f"Agent {LETTERS[j]}: last seen {'now' if step - e['step'] == 0 else f'{step - e[chr(115)+chr(116)+chr(101)+chr(112)]} steps ago'} at {e['pos']} facing {FACING[e['facing']]}; seen {e['count']} times this trial; {self._ate_text(j)}" + (f"; fired {e['zaps']} times in your view" if e["zaps"] else "") + (f"; sent you back to a spawn point {self.hit_me[j]} times" if self.hit_me.get(j) else "") + ".")
        return "\n".join(lines)

    def summary(self):
        parts = [f"{LETTERS[j]} never seen" if j not in self.seen and not self.hit_me.get(j) else
                 f"{LETTERS[j]} seen {self.seen.get(j, {}).get('count', 0)} steps, {self._ate_text(j)}"
                 + (f", sent you back to a spawn point {self.hit_me[j]} times" if self.hit_me.get(j) else "")
                 for j in range(self.n) if j != self.me]
        return "other agents: " + "; ".join(parts) + "."


class HarvestPlan(PlanLedger):
    KEEP_CLOSED = 8

    def on_decision(self, goal, step, pos, action, had_event, station_cells=()):
        if goal is not None and goal["type"] == "eat":
            g = dict(goal, type="move_to"); ok, chg = super().on_decision(g, step, pos, action, had_event); self.current["type"] = "eat"; return ok, chg
        return super().on_decision(goal, step, pos, action, had_event)

    def after_step(self, step, pos_after, fired=False, picked=False):
        g = self.current
        if not g:
            return
        if g["type"] == "eat":
            if tuple(pos_after) == tuple(g["cell"]):
                self._close("done", step)
            elif g["progress"] >= STALE_AFTER:
                self._close("stale", step)
            return
        if g["type"] == "wait":
            return                                  # waiting is the goal itself: it stays open until you set another one
        if g["type"] == "zap":                      # done the step the beam hits somebody; stale if you never get the shot
            if fired:
                self._close("done", step)
            elif g["progress"] >= STALE_AFTER:
                self._close("stale", step)
            return
        super().after_step(step, pos_after, fired, picked)

    def render(self, step):
        lines = []
        g = self.current
        if g:
            lines.append(f"Current goal #{g['idx']} (set step {g['set_step']}, {step - g['set_step']} steps ago): {g['type']} {tuple(int(x) for x in g['cell'])} — \"{g['why']}\"."
                         + (f" Distance now {g['dist']} (was {g['start_dist']})." if g["type"] in ("move_to", "eat") else ""))
        else:
            lines.append("No current goal (set one below).")
        if self.closed:
            groups = []                             # consecutive goals with the same type, cell and outcome are shown once
            for c in self.closed:
                key = (c["type"], tuple(int(x) for x in c["cell"]), c["status"])
                if groups and groups[-1][0] == key and c["status"] != "abandoned":
                    groups[-1][1].append(c)
                else:
                    groups.append((key, [c]))
            parts = []
            for (typ, cell, status), cs in groups:
                c = cs[-1]; ids = f"#{cs[0]['idx']}" + (f"-#{c['idx']}" if len(cs) > 1 else "")
                parts.append(f"{ids} {typ} {cell} — {status} at step {c['end_step']}" + (f" ({len(cs)} times)" if len(cs) > 1 else "")
                             + (f" (\"{c['end_why']}\")" if status == "abandoned" and c.get("end_why") else ""))
            lines.append("Earlier goals this trial: " + " · ".join(parts))
        return "\n".join(lines)

    @staticmethod
    def same(a, b):
        return a and b and a["type"].replace("eat", "move_to") == b["type"].replace("eat", "move_to") and tuple(a["cell"]) == tuple(b["cell"])


def parse_reply(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            obj = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            obj = None
    if not isinstance(obj, dict):
        return None
    try:
        action = int(obj["action"]); g = obj["goal"]
        goal = {"type": str(g["type"]), "cell": [int(g["cell"][0]), int(g["cell"][1])], "why": str(g.get("why", ""))[:80]}
        mode = str(obj["mode"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if not (0 <= action <= 7) or goal["type"] not in GOAL_TYPES or mode not in MODES:
        return None
    why = obj.get("why_this_move", obj.get("reasoning", ""))          # older replies used "reasoning"
    return {"reasoning": str(why)[:500], "mode": mode, "goal": goal, "action": action,
            "note": parse_note(obj.get("note"))}


# ---------------------------------------------------------------- the agent
class Agent:
    def __init__(self, provider, system, me, n_agents, history=5, max_tokens=800, strict=True, hues=None):
        self.provider, self.system, self.me = provider, system, me
        self.history_len, self.max_tokens, self.strict = history, max_tokens, strict
        self.ledger = HarvestLedger(hues=hues)
        # the facts are the ledger's; what the model works out about playing is its own, in its own words
        self.notebook = Notebook(summarise=P.build_summarise, summarise_schema=P.SUMMARISE_SCHEMA)
        self.note_feedback = None
        self.social = SocialLedger(me, n_agents)
        self.plan = HarvestPlan()
        self.recent = deque(maxlen=history)
        self.last_text, self.events = "", []
        self.parse_failures = 0; self.mode_counts = {m: 0 for m in MODES}
        self.earlier = []; self.trial_no = 0
        self.tree_hue = {}         # (r, c) -> hue of every tree cell you have ever seen (same layout every trial); grouped into
                                   # patches by position, since two patches can share a colour
        self._new_trial_state()

    def _new_trial_state(self):
        # per-trial memory (the orchard is re-laid at a trial boundary)
        # "full" (eaten at the hidden peak) is kept for analysis only; it must never reach the prompt
        self.tstats = {"points": 0.0, "apples": 0, "full": 0, "rot_seen": 0, "regrow_seen": 0, "by_stage": [0, 0, 0], "by_cs": {}}
        self.cells = {}            # (r, c) of a tree cell you saw being emptied -> {"hue", "left" (stage or ROT), "t0", "around", "last"}: waiting for it to regrow
        self.stage_watch = {}      # (r, c) of a tree cell with an apple -> {"stage", "start" (step it entered this stage, or None), "last"}
        self.cell_last = {}        # (r, c) of a tree cell -> {"hue", "t" (step you last saw it), "stage" (None = empty)}
        self.event_log = deque(maxlen=max(self.history_len, 1))   # (step, text) of events this trial, oldest first

    def trial_summary(self):
        ts = self.tstats
        cs = "; ".join(f"{HUE_NAMES[h]} {STAGE_WORD[st]} {n} ({pts:g} pts)" for (h, st), (n, pts) in sorted(ts["by_cs"].items()))
        return (f"Trial {self.trial_no + 1}: you ate {ts['apples']} apples for {ts['points']:.1f} points" + (f" ({cs})" if cs else "") +
                f"; you saw {ts['regrow_seen']} apples regrow and {ts['rot_seen']} rot.\nTrial {self.trial_no + 1}, " + self.social.summary())

    def new_trial(self):
        self.earlier.append(self.trial_summary())
        for rc, w in self.cells.items():        # cells still empty at the end of the trial: watched this long, never regrew
            if w["last"] > w["t0"]:
                self.ledger.record_open(w["hue"], w["left"], w["last"] - w["t0"], (w["around"], w["now"]))
        self.trial_no += 1; self._new_trial_state()
        self.social.reset(); self.plan.reset(); self.recent.clear(); self.last_text = ""
        self.events = ["NEW TRIAL: the orchard was re-laid with unripe apples (same layout) and you are back at a spawn point. "
                       "RULES YOU KNOW and EARLIER TRIALS are kept, and so is what you just wrote down; the step-by-step notes behind it are gone."]

    def consolidate(self, trial, outer):
        """Ask the model to rewrite its notes now that a trial has ended. Call this BEFORE new_trial(): the
        trial's own result (trial_summary) goes with the question, since it is being asked what paid off."""
        return self.notebook.consolidate(self.provider, self.system, trial, outer, self.max_tokens,
                                         outcome=self.trial_summary())

    @staticmethod
    def _cells_text(stages):
        k = [sum(1 for st in stages if st == x) for x in range(3)]; n = sum(k); e = len(stages) - n
        return f"{n} apple{'' if n == 1 else 's'} (unripe {k[0]}, ripe {k[1]}, over-ripe {k[2]}) and {e} empty cell{'' if e == 1 else 's'}"

    def patches_line(self, facts, now):
        in_view = set()
        for (r, c), h, st in [(rc, h, st) for rc, (h, st) in facts["apples"].items()] + [(rc, h, None) for rc, h in facts["empties"].items()]:
            self.tree_hue[(r, c)] = h; in_view.add((r, c))
            self.cell_last[(r, c)] = {"hue": h, "t": now, "stage": st}
        if not self.tree_hue:
            return ""
        groups = patches_of(self.tree_hue)                  # the patches you have seen, by position
        line = "Patches you have seen (same places every trial): " + "; ".join(patch_name(g, self.tree_hue) for g in groups) + "."
        # what you remember of the tree cells you cannot see right now, cell by cell: a glimpse of part of a patch
        # says nothing about the rest of it
        seen = []
        for g in groups:
            known = set(g); vis = known & in_view; hidden = known - vis
            if not hidden:
                continue                                    # all of it is in view: the NOW block already describes it
            head = f"{patch_name(g, self.tree_hue)} ({len(known)} tree cells seen so far" + (f", {len(vis)} in view now" if vis else "") + "): "
            mem = [self.cell_last[rc] for rc in hidden if rc in self.cell_last]
            if not mem:
                seen.append(head + (f"the {len(hidden)} out of view: " if vis else "") + "not seen yet this trial"); continue
            tmax = max(e["t"] for e in mem); recent = [e["stage"] for e in mem if e["t"] == tmax]; older = [e for e in mem if e["t"] < tmax]
            which = (f"all {len(known)}" if len(recent) == len(known) else f"{len(recent)} of them") if not vis else (
                "the rest" if len(recent) == len(hidden) else f"{len(recent)} of the {len(hidden)} out of view")
            parts = [f"at step {tmax} you saw {which}: {self._cells_text(recent)}"]
            if older:
                t0, t1 = min(e["t"] for e in older), max(e["t"] for e in older)
                parts.append(f"{'the other ' if len(recent) + len(older) == len(hidden) else ''}{len(older)} you last saw at step{'' if t0 == t1 else 's'} {t0 if t0 == t1 else f'{t0}-{t1}'}: {self._cells_text([e['stage'] for e in older])}")
            if len(mem) < len(hidden):
                parts.append(f"{len(hidden) - len(mem)} not seen yet this trial")
            seen.append(head + "; ".join(parts))
        if seen:
            line += "\nTree cells out of view, as you last saw them this trial: " + "; ".join(seen) + "."
        clocks = sorted(((e["start"], rc) for rc, e in self.stage_watch.items() if rc in facts["apples"] and e["start"] is not None), key=lambda x: (-x[0], x[1]))[:10]
        if clocks:
            line += "\nApples in view whose clock you know: " + "; ".join(f"{rc} {HUE_NAMES[facts['apples'][rc][0]]} {STAGE_WORD[facts['apples'][rc][1]]} since step {st}" for st, rc in clocks) + "."
        waiting = sorted(((w["t0"], rc, w) for rc, w in self.cells.items()), key=lambda x: (-x[0], x[1]))[:8]
        if waiting:
            line += "\nCells you are waiting on (you saw them emptied this trial and have not seen them regrow): " + "; ".join(
                f"{rc} {HUE_NAMES[w['hue']]} " + ("rotted" if w["left"] == ROT else f"eaten {STAGE_WORD[w['left']]}") + f" at step {t0}, " + (f"empty for {now - t0} steps so far" if w["last"] >= now else f"still empty when you last saw it at step {w['last']}") for t0, rc, w in waiting) + "."
        return line

    def decide(self, obs_text, facts, env, trial, outer, t, inner, zap_cd=0):
        ts = self.tstats
        progress = f"Trial {trial + 1} of {outer}, step {t} of {inner}. This trial: you ate {ts['apples']} apples for {ts['points']:.1f} points."
        older = [(st, e) for st, e in self.event_log if st < t]
        last = "\n".join(([self.last_text] if self.last_text else []) + (["Events: " + " ".join(self.events)] if self.events else [])
                         + ([f"Earlier events this trial (last {len(older)}): " + "; ".join(f"t{st} {e}" for st, e in older)] if older else []))
        if t == 0:
            for rc, (h, st) in facts["apples"].items():          # the orchard was just re-laid: every apple you see now started at step 0
                self.stage_watch.setdefault(rc, {"stage": st, "start": 0, "last": 0})
        seen = self.patches_line(facts, t)
        if seen:
            obs_text = obs_text.replace("\nMap:", "\n" + seen + "\nMap:", 1)
        live = [(w["hue"], w["left"], w["last"] - w["t0"], (w["around"], w["now"])) for w in self.cells.values() if w["last"] > w["t0"]]
        content = P.compose_user(progress, self.ledger.render(live), self.social.render(t), self.plan.render(t), last,
                                 "\n".join(self.recent), obs_text, actions_now(facts, env, zap_cd), earlier="\n".join(self.earlier),
                                 notes=self.notebook.render())
        messages = [{"role": "user", "content": content}]
        raw = self.provider.complete(self.system, messages, self.max_tokens, schema=ACTION_SCHEMA)
        reply = parse_reply(raw)
        if reply is None and not self.strict:
            retry = messages + [{"role": "assistant", "content": raw}, {"role": "user", "content": "That was not a valid reply. Reply with ONLY the JSON object."}]
            raw2 = self.provider.complete(self.system, retry, self.max_tokens, schema=ACTION_SCHEMA); reply = parse_reply(raw2); raw = raw + "\n---RETRY---\n" + raw2
        think = getattr(self.provider, "last_reasoning", "")
        if think:
            raw = raw + "\n---THINK---\n" + think
        had_event = bool(self.events)
        if reply is None:
            self.parse_failures += 1; reply = {"reasoning": "(invalid)", "mode": None, "goal": None, "action": STAY, "invalid": True}
        else:
            self.mode_counts[reply["mode"]] += 1
            res = self.notebook.apply(reply.get("note"), trial, t)
            reply["note_result"] = res
            reply["note_kept"] = bool(res["applied"] and res["op"] == "add")
            if res["op"] != "none" and not res["applied"]:
                # an edit that did not take is said so on the next step, or the model believes it did
                self.note_feedback = f"Your note edit ({res['op']} [{res['id']}]) was not applied: {res['why']}."
        reply["gen_tokens"] = getattr(self.provider, "last_tokens", None); reply["gen_seconds"] = getattr(self.provider, "last_seconds", None)
        reply["gen_reasoning_tokens"] = getattr(self.provider, "last_reasoning_tokens", 0)
        ok, chg = self.plan.on_decision(reply["goal"], t, facts["pos"], reply["action"], had_event)
        reply["consistent"], reply["changed_without_event"] = ok, chg
        return reply, content, raw

    def _emptied(self, rc, h, left, now, fb, fa):
        """A tree cell you saw losing its apple at step `now` (eaten at stage `left`, or ROT): start waiting for it to regrow.
        `around` = apples you could see hanging next to it (before or after the step), the env's regrowth neighbourhood."""
        near = {c for c in fa["apples"] if c != rc} | {c for c in fb["apples"] if c != rc and c not in fa["view"]}
        a = apples_around(rc, near)
        # around = at emptying; lo / hi / now = fewest, most and latest count while it waits (updated every step it is in view)
        self.cells[rc] = {"hue": h, "left": left, "t0": now, "around": a, "lo": a, "hi": a, "now": a, "last": now}

    def after_step(self, t, o, fb, fa, trial_end):
        now = t + 1                                            # fa is the state after step t
        # own eating
        if o["ate"]:
            h, st = o["ate_hue"], o["ate_stage"]
            self.ledger.record_eat(h, st, o["points"], fb["values"][h])
            self.tstats["apples"] += 1; self.tstats["points"] += o["points"]; self.tstats["full"] += int(o["points"] >= fb["values"][h] - 1e-6 and o["points"] > 0)
            self.tstats["by_stage"][st] += 1
            n, pts = self.tstats["by_cs"].get((h, st), (0, 0.0)); self.tstats["by_cs"][(h, st)] = (n + 1, pts + o["points"])
            self._emptied(tuple(o["moved_to"]), h, st, now, fb, fa); self.stage_watch.pop(tuple(o["moved_to"]), None)
        if trial_end:                                          # fa is already the re-laid orchard: diffing it against fb would log the re-lay as regrowth
            self.last_text = f"Last step ({t}): " + describe_outcome(o, fb["pos"])
            self.recent.append(recent_line(o, t, fb["pos"])); self.events = []
            return
        # regrowth / rot / others eating, from the view before vs after
        ev = []; regrew = {}; rotted = {}; eats = []
        common = fb["view"] & fa["view"]
        for rc in common:
            if rc in fb["empties"] and rc in fa["apples"]:                       # a new apple appeared where you saw an empty cell
                h = fb["empties"][rc]; w = self.cells.pop(rc, None); left = None
                if w is not None:
                    left = w["left"]; self.ledger.record_regrow(h, left, now - w["t0"], True, (w["lo"], w["hi"]))
                regrew[(h, left)] = regrew.get((h, left), 0) + 1; self.tstats["regrow_seen"] += 1
                self.stage_watch[rc] = {"stage": fa["apples"][rc][1], "start": now, "last": now}      # you saw it appear: its clock is exact
            elif rc in fb["apples"] and (rc in fa["empties"] or rc in fa["pos_of"]):    # the apple left
                h, st = fb["apples"][rc]
                if rc in fa["pos_of"]:
                    j = fa["pos_of"][rc]; eats.append((j, h, st)); self._emptied(rc, h, st, now, fb, fa)
                    ev.append(f"Agent {LETTERS[j]} ate a {HUE_NAMES[h]} {STAGE_WORD[st]} apple at {rc}.")
                elif st == NUM_RIPE_STAGES - 1:
                    self.ledger.rot[h] += 1; rotted[h] = rotted.get(h, 0) + 1; self.tstats["rot_seen"] += 1
                    e = self.stage_watch.get(rc)
                    if e and e["stage"] == st and e["last"] == now - 1 and e["start"] is not None:
                        self.ledger.record_ripen(st, now - e["start"])
                    self._emptied(rc, h, ROT, now, fb, fa)
                else:
                    self._emptied(rc, h, st, now, fb, fa)                           # eaten by someone you did not see
                self.stage_watch.pop(rc, None)
        # cells you are waiting on, seen empty again; cells that regrew while you were not looking
        for rc, w in list(self.cells.items()):
            if rc in fa["empties"]:
                w["last"] = now
                a = apples_around_if_seen(rc, fa)
                if a is not None:
                    w["now"] = a; w["lo"] = min(w["lo"], a); w["hi"] = max(w["hi"], a)
            elif rc in fa["apples"] and rc not in fb["view"]:
                self.ledger.record_regrow(w["hue"], w["left"], now - w["t0"], False, (w["lo"], w["hi"])); del self.cells[rc]
                self.stage_watch[rc] = {"stage": fa["apples"][rc][1], "start": None, "last": now}
        # ripening: follow every apple in view; a stage length is exact only if you saw both its start and its end
        for rc, (h, st) in fa["apples"].items():
            e = self.stage_watch.get(rc)
            if e is None:
                self.stage_watch[rc] = {"stage": st, "start": None, "last": now}
            elif e["stage"] == st:
                e["last"] = now
            elif e["stage"] < st:
                exact = e["last"] == now - 1 and st == e["stage"] + 1
                if exact and e["start"] is not None:
                    self.ledger.record_ripen(e["stage"], now - e["start"])
                self.stage_watch[rc] = {"stage": st, "start": now if exact else None, "last": now}
            else:
                self.stage_watch[rc] = {"stage": st, "start": None, "last": now}
        for rc in list(self.stage_watch):
            if rc in fa["view"] and rc not in fa["apples"]:
                del self.stage_watch[rc]
        self.last_text = f"Last step ({t}): " + describe_outcome(o, fb["pos"])
        self.recent.append(recent_line(o, t, fb["pos"]))
        self.plan.after_step(t, o["moved_to"], fired=bool(o.get("zap_hit")))
        self.social.update(now, fa["others"], set(o.get("zappers", [])), eats, o.get("zapped_by", []))
        came = sorted(set(fa["others"]) - set(fb["others"])); left = sorted(set(fb["others"]) - set(fa["others"]))
        if came: ev.append("Agent " + ", ".join(f"{LETTERS[j]} came into your view at {fa['others'][j][:2]}" for j in came) + ".")
        if left: ev.append("Agent " + ", ".join(LETTERS[j] for j in left) + " left your view.")
        if regrew: ev.append("Regrowth in your view: " + ", ".join(f"{n} {HUE_NAMES[h]}" + ((" (cell whose apple rotted)" if st == ROT else f" (cell eaten at {STAGE_WORD[st]})") if st is not None else " (you did not see that cell being emptied)") for (h, st), n in regrew.items()) + ".")
        if rotted: ev.append("Rotted in your view: " + ", ".join(f"{n} {HUE_NAMES[h]}" for h, n in rotted.items()) + ".")
        if o.get("zapped"): ev.append(f"You were ZAPPED: sent back to a spawn point and unable to act for the next {o.get('zap_wait', 0)} steps.")
        for e in ev:                                           # world events are kept for the trial; the blocked warning below is for this step only
            self.event_log.append((now, e))
        if o.get("bumps3"): ev.append("You have been blocked three steps in a row.")
        if self.note_feedback:
            ev.append(self.note_feedback); self.note_feedback = None
        self.events = [] if trial_end else ev


def describe_outcome(o, pos_before):
    a = o["action"]; names = {0: "north", 1: "south", 2: "east", 3: "west", 4: "turn left", 5: "turn right", 6: "zap", 7: "stay"}
    if o.get("invalid"):
        return "your reply was not valid JSON -> you stayed."
    s = f"you chose {a} ({names[a]})"
    if a in (0, 1, 2, 3):
        s += f" -> moved {tuple(pos_before)} -> {tuple(o['moved_to'])}" if tuple(o["moved_to"]) != tuple(pos_before) else " -> blocked, stayed"
    if o["ate"]:
        s += f" -> ATE a {HUE_NAMES[o['ate_hue']]} {STAGE_WORD[o['ate_stage']]} apple: +{o['points']:g} points"
    if a == ZAP:
        s += " -> zapped" + (f" agent {', '.join(LETTERS[j] for j in o.get('zap_hit', []))}" if o.get("zap_hit") else " nothing")
    return s + "."


def recent_line(o, t, pos_before):
    names = {0: "N", 1: "S", 2: "E", 3: "W", 4: "turnL", 5: "turnR", 6: "zap", 7: "stay"}
    s = f"t{t} {names[o['action']]}"
    if o["action"] in (0, 1, 2, 3): s += f" -> {tuple(o['moved_to'])}"
    if o["ate"]: s += f" ate {HUE_LETTER[o['ate_hue']]}{STAGE_LETTER[o['ate_stage']]} +{o['points']:g}"
    return s


class ScriptedProvider:
    model = "scripted"
    CYCLE = [0, 0, 2, 2, 1, 3, 0, 7, 2, 0]

    def __init__(self):
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}

    def complete(self, system, messages, max_tokens=400, schema=None):
        self.usage["calls"] += 1; a = self.CYCLE[self.usage["calls"] % len(self.CYCLE)]
        if schema is not None and "lessons" in schema.get("properties", {}):     # the trial-boundary rewrite
            return json.dumps({"lessons": [f"scripted lesson {self.usage['calls']}"], "todo": ["scripted plan"]})
        m = re.search(r"You are (?:agent \w, )?at \((\d+), (\d+)\)", messages[-1]["content"]); r, c = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        note = ({"op": "add", "id": 0, "text": f"scripted note {self.usage['calls']}"} if a == 7 else
                {"op": "replace", "id": 1, "text": f"scripted revision {self.usage['calls']}"} if a == 3 else {"op": "none", "id": 0, "text": ""})
        return json.dumps({"why_this_move": "scripted", "mode": "explore", "goal": {"type": "move_to", "cell": [max(r - 1, 0), c], "why": "scripted"},
                           "note": note, "action": a})


# ---------------------------------------------------------------- episode loop
def orchard_dead(state):
    """No apple anywhere and no rotted cell counting down: an eaten cell needs a hanging neighbour to regrow,
    so nothing can ever appear again this trial."""
    return not bool(is_apple(onp.array(state.grid)).any()) and not bool((onp.array(state.age) < 0).any())


def run_episode(env, agents, key, args, log, episode=0, recorder=None, on_step=None, init=None, max_steps=None, result=None):
    """on_step(trial, t, state, agents, obs, ks): optional hook called right before the agents decide at each step
    (snapshot tooling; it must not mutate anything). init: {"state": env State, "ks": step key} resumes mid-episode
    from a snapshot (the agents must carry the matching memory); max_steps stops after that many env steps;
    result (a dict) receives the final state, step key and step count. None of this changes an ordinary run."""
    n, inner, outer = env.num_agents, env.num_inner_steps, env.num_outer_steps
    if init is None:
        key, kr, ks = jax.random.split(key, 3)
        _, state = env.reset(kr)
    else:
        state = jax.tree_util.tree_map(jax.numpy.asarray, init["state"]); ks = jax.numpy.asarray(init["ks"])
    trees = tree_mask(env); hue_of_cell = hue_grid(env, state)
    speed_orders, pay_orders = decode_harvest_ruleset(state.rule_encoding)
    pay = [[STAGE_PAY[r] for r in PAY_ORDERS[int(o)]] for o in onp.array(pay_orders)]              # pay[h][stage] in points
    speed = [[float(REGROW_RATES[r]) for r in PAY_ORDERS[int(o)]] for o in onp.array(speed_orders)]   # speed[h][stage] = regrowth multiplier of a cell emptied at that stage
    truth = {"pay": pay, "speed": speed, "peaks": [p.index(max(p)) for p in pay], "fast_stage": [s.index(max(s)) for s in speed], "values": onp.array(state.values).tolist()}
    hp = present_hues(state)
    if init is None:
        print(f"[rules] hues {[HUE_NAMES[h] for h in hp]} | pay unripe/ripe/over-ripe {[truth['pay'][h] for h in hp]} | regrowth if eaten unripe/ripe/over-ripe {[[RATE_WORD[x] for x in truth['speed'][h]] for h in hp]}", flush=True)
    if args.reveal_rules or getattr(args, "reveal_regrowth", False):
        speed_words = [[RATE_WORD[x] for x in row] for row in truth["speed"]]
        for ag in agents:
            ag.ledger.given = (truth["pay"] if args.reveal_rules else None, speed_words)
    step = jax.jit(env.step_env); own_scale = 1.0 / n   # the environment pays an apple as n to the agent that ate it
    points_in = [[0.0] * outer for _ in range(n)]; apples_in = [[0] * outer for _ in range(n)]; bumps = [0] * n
    if init is not None:                                  # this trial's counters live in the agents' memory
        tr0 = int(state.outer_t)
        for i in range(n):
            points_in[i][tr0] = agents[i].tstats["points"]; apples_in[i][tr0] = agents[i].tstats["apples"]
    cur_trial, done = -1, False; n_steps = 0; env_done = False
    obs = [observe(env, state, i, trees, hue_of_cell) for i in range(n)]
    while not done:
        trial, t = int(state.outer_t), int(state.inner_t)
        if trial != cur_trial:
            if cur_trial >= 0:
                # every agent rewrites its notes (one extra call each) before the reset clears the trial
                with ThreadPoolExecutor(max_workers=n) as ex:
                    recs = list(ex.map(lambda i: agents[i].consolidate(cur_trial, outer), range(n)))
                for i, rec in enumerate(recs):
                    log.write(json.dumps({"episode": episode, "agent": i, "trial": cur_trial, "kind": "notebook", "notebook": rec}) + "\n")
                log.flush()
                for ag in agents: ag.new_trial()
            cur_trial = trial; obs = [observe(env, state, i, trees, hue_of_cell) for i in range(n)]
        if on_step is not None:
            on_step(trial, t, state, agents, obs, ks)
        zap_cd_now = onp.array(state.zap_cd).reshape(-1); zap_wait_now = onp.array(state.zap_wait).reshape(-1)
        def decide(i):
            if zap_wait_now[i] > 0:      # off the game after being zapped: the env ignores its action anyway
                return ({"reasoning": "(sent back by a beam; you cannot act yet)", "mode": None, "goal": None,
                         "action": STAY, "frozen": True}, "", "")
            return agents[i].decide(obs[i][0], obs[i][1], env, trial, outer, t, inner, int(zap_cd_now[i]))
        with ThreadPoolExecutor(max_workers=n) as ex:
            chosen = list(ex.map(decide, range(n)))
        raw_acts = [ch[0]["action"] for ch in chosen]
        fb = [obs[i][1] for i in range(n)]; locs_before = onp.array(state.agent_locs).copy()
        ks, k1 = jax.random.split(ks)
        _, state, rew, dones, info = step(k1, state, jax.numpy.array([RAW_TO_ENV[a] for a in raw_acts]))
        n_steps += 1
        env_done = bool(dones["__all__"])
        done = env_done or (max_steps is not None and n_steps >= max_steps)
        trial_ended = int(state.inner_t) == 0 and int(state.outer_t) != trial
        locs_after = onp.array(state.agent_locs); own = onp.array(info["original_rewards"]).reshape(-1) * own_scale
        eaten = onp.array(info["eaten"]).reshape(-1); eh = onp.array(info["eaten_hue"]).reshape(-1); es = onp.array(info["eaten_stage"]).reshape(-1)
        obs = [observe(env, state, i, trees, hue_of_cell) for i in range(n)]
        if recorder is not None:
            recorder.set_caption(f"A0 {chosen[0][0].get('mode')} {chosen[0][0]['action']} - {chosen[0][0]['reasoning'][:70]}")
            recorder(state, {"trial": trial, "t": t, "held": 0, "crafted": 0, "reward": float(own.sum()),
                             "fired_agents": [i for i in range(n) if raw_acts[i] == ZAP]}, trial_ended)
        hit_by_shooter = {i: [j for j in range(n) if j != i and raw_acts[i] == ZAP
                               and (int(locs_before[j][0]), int(locs_before[j][1])) in [rc for rc, _ in fb[i]["beam"]]] for i in range(n)}
        hit_by = {j: [i for i in range(n) if j in hit_by_shooter[i]] for j in range(n)}      # victim -> who sent it back
        for i, (reply, prompt, raw) in enumerate(chosen):
            fa = obs[i][1]
            moved = raw_acts[i] in (0, 1, 2, 3); blocked = moved and tuple(locs_before[i][:2]) == tuple(locs_after[i][:2])
            bumps[i] = bumps[i] + 1 if blocked else 0
            jumped = not trial_ended and abs(int(locs_after[i][0]) - int(locs_before[i][0])) + abs(int(locs_after[i][1]) - int(locs_before[i][1])) > 1
            zap_hit = hit_by_shooter[i]
            o = {"action": raw_acts[i], "invalid": reply.get("invalid", False), "frozen": reply.get("frozen", False),
                 "zap_wait": int(onp.array(state.zap_wait).reshape(-1)[i]), "zap_cd": int(onp.array(state.zap_cd).reshape(-1)[i]),
                 "moved_from": [int(x) for x in locs_before[i][:2]], "moved_to": [int(x) for x in locs_after[i][:2]],
                 "facing_after": FACING[int(locs_after[i][2])], "bumps": int(blocked), "bumps3": bumps[i] >= 3,
                 "ate": bool(eaten[i]), "ate_hue": int(eh[i]), "ate_stage": int(es[i]), "points": float(own[i]),
                 "full": bool(eaten[i]) and float(own[i]) >= fb[i]["values"][int(eh[i])] - 1e-6 and float(own[i]) > 0,
                 "zapped": bool(jumped), "zap_hit": zap_hit, "zapped_by": hit_by[i], "zappers": [j for j in range(n) if raw_acts[j] == ZAP and j in fa["others"]],
                 "result": "trial ended" if trial_ended else "done"}
            agents[i].after_step(t, o, fb[i], fa, trial_ended)
            points_in[i][trial] += float(own[i]); apples_in[i][trial] += int(eaten[i])
            log.write(json.dumps({"episode": episode, "agent": i, "trial": trial, "t": t, "action": reply, "outcome": o,
                                  "outcome_text": agents[i].last_text, "raw": raw, "prompt": prompt}) + "\n")
        if getattr(args, "early_stop", True) and not done and not trial_ended and orchard_dead(state):
            # evaluation shortcut (not a game rule, the agents are not told): nothing can happen any more this trial,
            # so play the remaining steps as STAY without asking the model. Same key splits as real steps, so a
            # replay of the log (which marks the skipped range) reproduces the run exactly.
            stay = jax.numpy.array([RAW_TO_ENV[STAY]] * n); last = t
            while True:
                ks, k1 = jax.random.split(ks)
                _, state, rew, dones, info = step(k1, state, stay); last += 1; n_steps += 1
                assert float(onp.array(info["original_rewards"]).sum()) == 0.0
                done = env_done = bool(dones["__all__"])
                if done or (int(state.inner_t) == 0 and int(state.outer_t) != trial):
                    break
            log.write(json.dumps({"episode": episode, "event": "fast_forward", "trial": trial, "from_t": t + 1, "to_t": last}) + "\n")
            done = done or (max_steps is not None and n_steps >= max_steps)
            print(f"[early stop] trial {trial + 1}: orchard empty after step {t}, skipped steps {t + 1}-{last}", flush=True)
            if recorder is not None:
                recorder.flush(trial)
            obs = [observe(env, state, i, trees, hue_of_cell) for i in range(n)]
        log.flush()
    if recorder is not None:
        recorder.flush(cur_trial)
    per_agent = [[{"return": points_in[i][tr], "apples": apples_in[i][tr]} for tr in range(outer)] for i in range(n)]
    team = [sum(points_in[i][tr] for i in range(n)) for tr in range(outer)]
    if result is not None:
        result["state"], result["ks"], result["steps"], result["env_done"] = state, ks, n_steps, env_done
    return per_agent, team, truth


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["vllm", "openai", "scripted", "anthropic"], default="vllm")
    ap.add_argument("--model", default="m"); ap.add_argument("--models", default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:8550/v1"); ap.add_argument("--base-urls", default=None)
    ap.add_argument("--temperature", type=float, default=0.2); ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--reasoning-effort", default=None); ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--think-budget", type=int, default=None); ap.add_argument("--think-effort", default=None, choices=["low", "medium", "xhigh"])
    ap.add_argument("--history", type=int, default=5, help="recent steps shown (and events kept this trial); 5 like cleanup since 2026-09-21, the runs before used 15"); ap.add_argument("--lenient", action="store_true")
    ap.add_argument("--reveal-rules", action="store_true", help="ablation: tell the agents the peak stage and regrowth speed of every colour")
    ap.add_argument("--ripen-range", default="15,20", help="steps per ripeness stage, drawn per apple from this closed range (before 2026-09-19: 25,25)")
    ap.add_argument("--rot-range", default="2,5", help="steps from rot to the cell regrowing, drawn per rot (before 2026-09-19: 5,5)")
    ap.add_argument("--rates", default="2,1,0.5", help="fast / normal / slow regrowth multipliers (before 2026-09-19: 2,1,0.25)")
    ap.add_argument("--respawn-wait", type=int, default=12,
                    help="steps a zapped agent spends unable to act (12 since 2026-09-24, user's choice for a beam that is a real deterrent; Melting Pot commons_harvest__open uses 4)")
    ap.add_argument("--reveal-regrowth", action="store_true", help="ablation: tell the agents how fast a cell regrows after being eaten at each stage, but not what an apple pays")
    ap.add_argument("--no-early-stop", dest="early_stop", action="store_false", help="evaluation shortcut off: ask the model on every step even after the orchard is empty and cannot regrow")
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--inner", type=int, default=500); ap.add_argument("--outer", type=int, default=10); ap.add_argument("--ripen", type=int, default=25)
    ap.add_argument("--on-train", action="store_true"); ap.add_argument("--seed", type=int, default=906); ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--tag", default="harv"); ap.add_argument("--log-dir", default="logs/llm"); ap.add_argument("--out-dir", default="logs/llm")
    ap.add_argument("--gif-dir", default=None); ap.add_argument("--gif-trials", default="0"); ap.add_argument("--gif-every", type=int, default=1); ap.add_argument("--gif-fps", type=int, default=8)
    return ap


def main():
    a = build_parser().parse_args()

    def env_for(ep):                         # every episode its own map, fixed by its seed
        from opensocialjax.environments.open_harvest.layouts_orig6 import pick_layout
        layout = pick_layout(a.seed + ep, a.on_train)
        return make_env(a.agents, a.inner, a.outer, a.ripen, a.on_train, layout, a.respawn_wait, ripen_range=[int(x) for x in a.ripen_range.split(",")],
                        rot_range=[int(x) for x in a.rot_range.split(",")], rates=[float(x) for x in a.rates.split(",")]), layout
    env, layout = env_for(0)
    if a.provider == "scripted":
        providers = [ScriptedProvider() for _ in range(a.agents)]
    elif a.provider == "anthropic":           # Claude on the Anthropic SDK (key in ~/.config/anthropic/key); --reasoning-effort = its effort
        from llm_policy.anthropic_provider import AnthropicProvider
        providers = [AnthropicProvider(a.model, effort=a.reasoning_effort or "medium") for _ in range(a.agents)]
    else:
        models = (a.models or a.model).split(","); urls = (a.base_urls or a.base_url).split(","); api_key = "EMPTY"
        if a.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY") or open(os.path.expanduser("~/.config/openai/key")).read().strip(); urls = ["https://api.openai.com/v1"]
        extra = ({"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": a.think_effort}} if a.think_effort else
                 {"chat_template_kwargs": {"enable_thinking": False}} if a.no_think else None)
        providers = [ChatProvider(models[i % len(models)].strip(), urls[i % len(urls)].strip(), api_key, a.temperature,
                                  extra_body=extra, reasoning_effort=a.reasoning_effort, think_budget=a.think_budget) for i in range(a.agents)]
    os.makedirs(a.log_dir, exist_ok=True); os.makedirs(a.out_dir, exist_ok=True)
    log_path = os.path.join(a.log_dir, f"{a.tag}_decisions.jsonl")
    print(f"{providers[0].model} | harvest prompt | {a.episodes} episodes of {a.outer}x{a.inner} | log {log_path}", flush=True)
    results = []
    with open(log_path, "w") as log:
        for ep in range(a.episodes):
            if ep > 0:
                env, layout = env_for(ep)
            print(f"episode {ep}: map {layout} of the {'training' if a.on_train else 'held-out'} pool", flush=True)
            hues = present_hues(initial_state(env, a.seed, ep))        # the colours drawn for this meta-episode (ledger order only; the prompt never lists them)
            systems = [P.build_system(a.agents, a.inner, a.outer, a.ripen, me=i, respawn_wait=a.respawn_wait) for i in range(a.agents)]   # each agent is told its own letter; the zap freeze is the env's
            agents = [Agent(providers[i], systems[i], i, a.agents, history=a.history, max_tokens=a.max_tokens, strict=not a.lenient, hues=hues) for i in range(a.agents)]
            if ep == 0:
                open(os.path.join(a.log_dir, f"{a.tag}_system.txt"), "w").write(systems[0])
            recorder = None
            if a.gif_dir and ep == 0:
                from llm_policy.record import GifRecorder
                recorder = GifRecorder(env, a.gif_dir, a.tag, trials=[int(x) for x in a.gif_trials.split(",")], every=a.gif_every, fps=a.gif_fps)
            t0 = time.time()
            try:
                per_agent, team, truth = run_episode(env, agents, jax.random.PRNGKey(a.seed + ep), a, log, ep, recorder)
            except BaseException:
                if recorder is not None and recorder.frames:
                    recorder.flush(recorder.current or 0)
                raise
            nb = [{"notes": ag.notebook.n_notes, "edits": len(ag.notebook.edits),
                   "rewrites_ok": sum(r["ok"] for r in ag.notebook.history), "rewrites_failed": sum(not r["ok"] for r in ag.notebook.history),
                   "lessons_now": list(ag.notebook.lessons), "todo_now": list(ag.notebook.todo)} for ag in agents]
            results.append({"per_agent": per_agent, "team": team, "truth": truth, "notebook": nb, "pool": POOL, "layout": layout,
                            "usage": [dict(getattr(p, "usage", {})) for p in providers]})   # tokens and seconds per agent, for the cost table
            print(f"episode {ep}: TEAM points/trial {[round(x, 1) for x in team]} | apples/trial per agent {[[x['apples'] for x in pa] for pa in per_agent]} | "
                  f"invalid {[ag.parse_failures for ag in agents]} | goal changes w/o event {sum(ag.plan.changes_without_event for ag in agents)} | "
                  f"modes {[ag.mode_counts for ag in agents]} | notes {[x['notes'] for x in nb]}, edits {[x['edits'] for x in nb]}, "
                  f"rewrites ok {sum(x['rewrites_ok'] for x in nb)}/{sum(x['rewrites_ok'] + x['rewrites_failed'] for x in nb)} | "
                  f"{sum(p.usage['calls'] for p in providers)} calls, {time.time() - t0:.0f}s", flush=True)
            for ag in agents:
                print(f"  A{ag.me} ledger:\n    " + ag.ledger.render().replace("\n", "\n    "), flush=True)
    json.dump({"args": vars(a), "results": results}, open(os.path.join(a.out_dir, f"llm_{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
