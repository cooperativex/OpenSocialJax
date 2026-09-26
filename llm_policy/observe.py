"""Serialise the agent's 11x11 egocentric window as text.

Same window the recurrent policy sees (9 cells ahead, 1 behind, 5 to each
side — see ``_get_obs_point``), same content (cell kind + colour, own
position, held tool) — rendered in words instead of one-hot channels, so
an LLM can read it. The hidden rule is never in it.
"""
from __future__ import annotations

import numpy as onp

from opensocialjax.environments.open_cleanup.open_cleanup import (
    ATTR_HUE_NAMES, ATTR_SHADE_NAMES, DIRT_BASE, NUM_COLORS, NUM_DIRT_TYPES, NUM_SHADES,
    NUM_TOOLS, TOOL_BASE, Items,
)


def colour_name(env, c):
    """'7' — or '7 (green-mid)' in the attribute variant, where the hue and
    shade are what the rule is written in."""
    c = int(c)
    if getattr(env, "attr_rules", False):
        return f"{c} ({ATTR_HUE_NAMES[c // NUM_SHADES]}-{ATTR_SHADE_NAMES[c % NUM_SHADES]})"
    return str(c)

# world-frame unit vectors per facing (matches STEP in the environment)
_FWD = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
_COMPASS = {0: "south (towards the orchard)", 1: "east",
            2: "north (towards the river)", 3: "west"}
AHEAD, BEHIND, SIDE = 9, 1, 5


def _rel_compass(a, b):
    """(a, b) = (cells north, cells east) -> compass phrase (north-up mode)."""
    parts = []
    if a > 0: parts.append(f"{a} north")
    elif a < 0: parts.append(f"{-a} south")
    if b > 0: parts.append(f"{b} east")
    elif b < 0: parts.append(f"{-b} west")
    return ", ".join(parts) if parts else "here"


def _rel(a, b):
    """(a, b) = (cells ahead, cells to the right) -> short phrase."""
    parts = []
    if a > 0:
        parts.append(f"{a} ahead")
    elif a < 0:
        parts.append(f"{-a} behind")
    if b > 0:
        parts.append(f"{b} right")
    elif b < 0:
        parts.append(f"{-b} left")
    return ", ".join(parts) if parts else "here"


# Ablation switch (--orig-hand): the pre-2026-09-02 hand text, which named
# a crafted tool only as "a CRAFTED tool (which one, only firing tells)".
ORIG_HAND_TEXT = False


def held_text(env, held, pickups=None):
    """Attr-craft: what the observation says about the hand. A crafted tool is
    named by the pickups that made it (the agent's own record), never by its
    product id — what it does is for the agent's firing log to say."""
    held = int(held)
    if held >= env.num_tools:
        if ORIG_HAND_TEXT:
            return "a CRAFTED tool (which one, only firing tells)"
        picks = [int(x) for x in (pickups if pickups is not None else []) if int(x) >= 0]
        if picks:
            return f"the tool crafted by your pickups {picks}"
        return "a CRAFTED tool"
    return f"base tool {held} (does nothing to waste by itself)"


_COMPASS_SHORT = {0: "SOUTH", 1: "EAST", 2: "NORTH", 3: "WEST"}


def describe(env, state, agent=0, north_up=False, ego_abs=False):
    """Return (text, facts) for one agent's current window.

    Default: the policy's own window (9 ahead, 1 behind, 5 to each side,
    turned so the top is the way the agent faces) with relative phrases.
    ``north_up``: a symmetric 5/5 window in world frame (top = north/river)
    with absolute (row, col) positions — the raw-action mode's first view.
    ``ego_abs``: the policy's window (turned, 9/1/5) but with absolute
    (row, col) positions and labelled axes — the raw-action mode's view when
    it is to see exactly what the trained policy sees."""
    grid = onp.asarray(state.grid)
    loc = onp.asarray(state.agent_locs)[agent]
    r0, c0, d = int(loc[0]), int(loc[1]), int(loc[2])
    if north_up:
        fr, fc = (-1, 0)                 # picture "up" = north (row-)
        rr, rc = (0, 1)                  # picture "right" = east (col+)
        ahead, behind = SIDE, SIDE
    else:
        fr, fc = _FWD[d]
        # the agent's right: the env turns left with d+1 (south -> east ->
        # north -> west), so right is d-1 — (d+1) would mirror the picture
        rr, rc = _FWD[(d - 1) % 4]
        ahead, behind = AHEAD, BEHIND
    abs_pos = north_up or ego_abs        # positions as absolute (row, col)
    rel = _rel_compass if north_up else _rel
    def _abs(a, b):
        # window offsets (ahead, right) -> absolute (row, col)
        return (r0 + a * fr + b * rr, c0 + a * fc + b * rc)
    H, W = grid.shape
    held = int(onp.asarray(state.held_tool)[agent])
    attr = bool(getattr(env, "attr_rules", False))

    waste, stations, apples, others = {}, {}, [], []
    water = 0
    rows = []
    for a in range(ahead, -behind - 1, -1):          # top of the picture = far ahead (north-up: north)
        cells = []
        for b in range(-SIDE, SIDE + 1):
            r, c = r0 + a * fr + b * rr, c0 + a * fc + b * rc
            if not (0 <= r < H and 0 <= c < W):
                cells.append("##"); continue
            code = int(grid[r, c])
            if a == 0 and b == 0:
                cells.append("ME")
            elif code == int(Items.wall):
                cells.append("##")
            elif code == int(Items.apple):
                cells.append("AP"); apples.append((a, b))
            elif code in (int(Items.river), int(Items.potential_dirt)):
                cells.append("~~"); water += 1
            elif DIRT_BASE <= code < DIRT_BASE + NUM_DIRT_TYPES:
                col = code - DIRT_BASE
                cells.append(f"{ATTR_HUE_NAMES[col // NUM_SHADES][0].upper()}{col % NUM_SHADES}"
                             if attr else f"W{col:X}")
                waste.setdefault(col, []).append((a, b))
            elif TOOL_BASE <= code < TOOL_BASE + NUM_COLORS:
                t = code - TOOL_BASE
                cells.append(f"T{t}")
                stations.setdefault(t, []).append((a, b))
            elif code >= len(Items):
                cells.append("AG"); others.append((a, b))
            elif code == int(Items.clean_beam):
                cells.append("**")
            else:
                cells.append("..")
        if north_up:
            rows.append(f"r{r0 - a:>2} " + " ".join(cells))
        elif ego_abs:
            # label each picture row by its absolute coordinate along the
            # facing axis (row number when facing N/S, column when E/W)
            lab = f"r{r0 + a * fr:>2}" if fc == 0 else f"c{c0 + a * fc:>2}"
            rows.append(lab + " " + " ".join(cells))
        else:
            rows.append(" ".join(cells))

    def nearest(lst):
        return min(lst, key=lambda ab: abs(ab[0]) + abs(ab[1]))

    if getattr(env, "craft_rules", False):
        picks = [int(x) for x in onp.asarray(state.pickups)[agent] if int(x) >= 0]
        if getattr(env, "craft_show_tool", False):
            crafted = bool(onp.asarray(state.crafted)[agent])
            lines = [f"You face {_COMPASS[d]}. Cleaning tool: {'OBTAINED' if crafted else 'NOT obtained'}. "
                     f"Your last pickups (oldest first): {picks if picks else 'none'}."]
        elif getattr(env, "attr_craft", False):
            # name the hand every step, by the pickups that made it (never by
            # what it does): the pickup pair alone left the agent recomputing
            # what it holds, and losing track of it after a stray pickup.
            lines = [f"You face {_COMPASS[d]}. You hold {held_text(env, held, picks)}. "
                     f"Your last pickups (oldest first): {picks if picks else 'none'}."]
        else:
            lines = [f"You face {_COMPASS[d]}. Your last pickups (oldest first): {picks if picks else 'none'}."]
    else:
        lines = [f"You face {_COMPASS[d]}. You hold tool {held}."]
    if abs_pos:
        dr = onp.asarray(state.potential_dirt_and_dirt_locs)
        rows_river = (int(dr[:, 0].min()), int(dr[:, 0].max())) if dr.size else (0, 0)
        orch = onp.asarray(getattr(env, "POTENTIAL_APPLE", []))
        rows_orch = (int(orch[:, 0].min()), int(orch[:, 0].max())) if orch.size else None
        # explicit_pickup: the station under your feet is hidden by ME on the
        # map, and it is exactly where the pick-up action works -> say it.
        under = int(env._tool_grid[r0, c0]) if getattr(env, "explicit_pickup", False) else -1
        if under >= 0:
            lines.insert(0, f"You are STANDING ON tool station T{under} (tool {under}).")
        lines.insert(0, f"You are at (row {r0}, col {c0}); row 0 is the NORTH edge. "
                        f"The river (waste) is rows {rows_river[0]}-{rows_river[1]}"
                        + (f"; the orchard is rows {rows_orch[0]}-{rows_orch[1]}." if rows_orch else "."))
    hue_counts = {}
    if waste:
        if attr:
            for c, v in waste.items():
                hue_counts[c // NUM_SHADES] = hue_counts.get(c // NUM_SHADES, 0) + len(v)
            # dominant hue first, then by shade — the order the rule is written in
            items = sorted(waste.items(), key=lambda kv: (-hue_counts[kv[0] // NUM_SHADES], kv[0]))
            dom_h = max(hue_counts, key=hue_counts.get)
            by_shade = [sum(len(v) for c, v in waste.items() if c // NUM_SHADES == dom_h and c % NUM_SHADES == s)
                        for s in range(NUM_SHADES)]
            lines.append(f"Dominant hue in view: {ATTR_HUE_NAMES[dom_h]} ("
                         + ", ".join(f"{ATTR_SHADE_NAMES[s]} {n}" for s, n in enumerate(by_shade)) + ").")
        else:
            items = sorted(waste.items(), key=lambda kv: -len(kv[1]))
        if abs_pos:
            lines.append("Waste in view: " + "; ".join(
                f"colour {colour_name(env, c)}: {len(v)} cells (nearest at {_abs(*nearest(v))})" for c, v in items))
        else:
            lines.append("Waste in view: " + "; ".join(
                f"colour {colour_name(env, c)}: {len(v)} cells (nearest {rel(*nearest(v))})" for c, v in items))
    else:
        lines.append("Waste in view: none.")
    if stations:
        if abs_pos:
            lines.append("Tool stations in view: " + "; ".join(
                f"tool {t} at {_abs(*nearest(v))}" for t, v in sorted(stations.items())))
        else:
            lines.append("Tool stations in view: " + "; ".join(
                f"tool {t} ({rel(*nearest(v))})" for t, v in sorted(stations.items())))
    else:
        lines.append("Tool stations in view: none.")
    n_waste = sum(len(v) for v in waste.values())
    if n_waste + water:
        lines.append(f"River in view: {n_waste} waste cells vs {water} clean water cells "
                     f"({100 * n_waste / (n_waste + water):.0f}% polluted here).")
    else:
        lines.append("River: NOT in view (you are away from it; its state is unknown from here).")
    lines.append(f"Apples in view: {len(apples)}" + ((f" (nearest at {_abs(*nearest(apples))})" if abs_pos
                                                       else f" (nearest {rel(*nearest(apples))})") if apples else "."))
    if others:
        lines.append(f"Other agents in view: {len(others)}.")
    if attr:
        if north_up:
            hdr = "Map (top = NORTH / the river, bottom = SOUTH / the orchard, left = WEST, ME = you; "
        elif ego_abs:
            hdr = (f"Map, drawn the way YOU face: top = {_COMPASS_SHORT[d]} (ahead of you), "
                   f"bottom = {_COMPASS_SHORT[(d + 2) % 4]} (behind), left = {_COMPASS_SHORT[(d + 1) % 4]}, "
                   f"right = {_COMPASS_SHORT[(d - 1) % 4]}; you see 9 cells ahead, 1 behind, 5 to each side; "
                   + ("rows are labelled with their row number, the header with column numbers; "
                      if fc == 0 else "rows are labelled with their COLUMN number, the header with ROW numbers; ")
                   + "ME = you; ")
        else:
            hdr = "Map (top = far ahead, ME = you; "
        if north_up:
            W_ = grid.shape[1]
            rows.insert(0, "    " + " ".join(f"{c0 + b:>2}" if 0 <= c0 + b < W_ else "  "
                                             for b in range(-SIDE, SIDE + 1)))
        elif ego_abs:
            H_, W_ = grid.shape
            # header = absolute coordinate along the side axis (col when facing N/S, row when E/W)
            side = [(c0 + b * rc, W_) if fc == 0 else (r0 + b * rr, H_) for b in range(-SIDE, SIDE + 1)]
            rows.insert(0, "    " + " ".join(f"{v:>2}" if 0 <= v < n else "  " for v, n in side))
        lines.append(hdr + "waste = hue letter + shade digit, e.g. G2 = green-dark "
                     "(R/Y/G/C/P = red/yellow/green/cyan/purple, 0 light / 1 mid / 2 dark; colour code = hue*3 + shade); "
                     "T<n> tool station, AP apple, ~~ water, ## wall, .. ground):")
    else:
        lines.append("Map (top = far ahead, ME = you; W<hex> waste colour, T<n> tool station, "
                     "AP apple, ~~ water, ## wall, .. ground):")
    lines.extend(rows)
    facts = {"held": held, "facing": d, "waste_counts": {c: len(v) for c, v in waste.items()},
             "stations": sorted(stations), "apples": len(apples), "water": water,
             "hue_counts": hue_counts}
    return "\n".join(lines), facts
