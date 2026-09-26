"""OpenCleanup prompt, written out in full.

SYSTEM  -- S1 objective, S2 the world, S3 how to read the observation, S4 the
           output format with two examples, S5 the notes.  Sent once per run
           per agent, filled from {me}, {n_agents}, {letters}, {inner}, {outer}.
           Rewards are individual only (the environment has no common-reward mode).
USER    -- ten blocks (+ EARLIER TRIALS from the second trial on) rebuilt every
           step by compose_user(); the block headers live here, the block
           contents come from open_cleanup_policy.
SUMMARISE
        -- a separate call made at each trial boundary, and the only one that is
           not a move: the model is shown its notes and rewrites them, and the
           answer becomes its notes for the next trial.

Who works out what:

  the harness   what each shot proved. Evidence records every pick-up order this
                agent has fired with and what those shots did, and shows it under
                RULES YOU KNOW for the whole episode. The model never has to
                write a shot down, or join a shot to the hand that fired it.
  the model     how to play. Which order of doing things pays off, what a tool
                swap costs, how far the river has to come down before apples
                appear, what the other agents are up to. No ledger can compute
                any of that, so it goes in the model's own notes, one line per
                step, rewritten into a short list at every trial boundary.

What the prompt states: the colour codes, the three things a shot can do, the
stations, how a pick-up slides the window, that a working tool is two pick-ups
in order, the beam's shape, movement, and that apples need a clean river. What
it does not state: which order makes which tool (found by trying, then kept by
the ledger), where to stand to hit a cell, how many pick-ups a recipe still is
from the window in hand, and what to do first.

Everything the model reads is in this file or produced by open_cleanup_policy.py.
"""

# ---------------------------------------------------------------- system prompt
SYSTEM = """You are agent {me}, one of {n_agents} agents ({letters}) in a grid world called Clean Up.

1. YOUR OBJECTIVE
1.1 Score = apples eaten. Only the apples YOU eat count for you. Cleaning the river helps everyone; apples go to whoever gets there first.
1.2 Apples grow only in the orchard, and only while the river is clean enough.
1.3 A trial lasts {inner} steps. Then the map is re-laid in the same layout (stations stay where they were): the river is fully polluted again, your hand is empty, apples are gone. The recipes (see 2.6) and what each working tool does stay the same for all {outer} trials of the episode.
1.4 You are told only what YOU cleared and ate. You never see other agents' scores, hands or windows.

2. THE WORLD
2.1 Waste cells have a HUE (red, yellow, green, cyan, purple) and a SHADE (light, mid, dark). Colour code = hue*3 + shade, e.g. 6 = green-light, 7 = green-mid, 8 = green-dark.
2.2 Firing your beam at a waste cell does exactly one of: CLEARS it to water, makes it ONE SHADE LIGHTER (dark -> mid, mid -> light), or NOTHING.
2.3 A WORKING tool is made by picking up two base tools in order; a base tool on its own does nothing to waste. Which pick-up order makes which working tool, and which waste that tool acts on, you find out by trying it — RULES YOU KNOW then keeps it for you.
2.4 There are 4 kinds of BASE tool stations (tools 0-3), in a row between the river and the orchard, several of each.
2.5 PICK UP = action 8 while standing on a station. Walking onto or across a station does nothing to your hand.
2.6 You have one hand and a WINDOW = your last two pick-ups, oldest first. Every pick-up pushes the oldest one out: window [2, 0], pick up 3 -> window [0, 3]. Each working tool has a hidden RECIPE, an ordered pair such as [2, 0] = pick up tool 2, then tool 0. The moment your window equals a recipe, that working tool is in your hand; the next pick-up slides the window and you hold a plain base tool again. No two working tools have the same recipe.
2.7 The observation names what is in your hand only by the pick-ups that made it, never by what it is or does — it does not even tell you whether those two pick-ups were a recipe at all. Firing is what tells you.
2.8 Firing: the beam covers the cell you FACE, the cell beyond it, and the two cells diagonally beside the faced cell.
2.9 Moves are absolute compass moves and do not change your facing; only turning does. A move into a wall or another agent is blocked. Walking onto an apple eats it (+1). Waste slowly reappears.

3. HOW TO READ YOUR TURN
3.1 Positions are (row, col): row 0 is the NORTH edge (river) and grows southward; col 0 is the WEST edge and grows eastward. North = row-1, south = row+1, east = col+1, west = col-1.
3.2 You see an 11x11 window that turns with you: 9 cells ahead of you, 1 cell behind and 5 to each side, so turning changes what you see. The map shows exactly those cells, always drawn north-up with columns labelled on top and rows on the left (ME sits one cell in from the side behind you); cells beyond the edge of the world are ##. ME = you, @{other} = agent {other} (every other agent is shown by its letter), T<n> = station of tool n, AP = apple, ~~ = water, ## = wall, .. = ground, waste = hue letter + shade digit (G2 = green-dark).
3.3 RULES YOU KNOW: every pick-up order you have fired with and what those shots turned out to do, worked out and kept for you — you never have to write a shot down yourself. YOUR NOTES: yours, and the only thing here that is — what you have worked out about how to PLAY, and what you mean to do next. Nothing in it is checked or filled in for you. EARLIER TRIALS: one line per finished trial with what you did and what you saw of the others; use it to do better this trial. OTHER AGENTS: only what you have seen in your own view: where each agent was, whether it was firing, and what its shots did to the waste cells you could see. YOUR PLAN: your current goal and the goals you set before, kept for you. LAST STEP: what your last action did, and events that happened in your view. RECENT: your last few steps. ACTIONS NOW: what you can do right now; fire shows what the beam would hit.
3.4 You know only what you see: other agents and the river exist beyond your view, but you get no information about them.

4. HOW TO ANSWER
4.1 Reply with ONE JSON object and nothing else:
{"why_this_move": "<up to 500 characters>",
 "mode": "craft" | "clean" | "harvest" | "explore",
 "goal": {"type": "move_to" | "fire" | "pick_up" | "wait", "cell": [row, col], "why": "<up to 80 characters>"},
 "note": {"op": "none" | "add" | "replace" | "delete", "id": <entry number to replace or delete, else 0>, "text": "<the entry to add, or the new wording for [id]; \\"\\" otherwise>"},
 "action": <one integer from ACTIONS NOW>}
4.2 mode = what you are working on: craft (getting a working tool), clean (firing at waste), harvest (going for apples), explore (testing an unknown pair).
4.3 goal = the sub-goal your action serves. Keep the SAME goal from step to step until it is done or an event makes it pointless; if you change it, say why in "why". For fire / pick_up / wait, cell = where you do it.
4.4 Example, keeping the goal:
{"why_this_move": "Both shots I have fired since my window became [2, 0] turned a G2 cell into G1, which nothing I held before did. Two more G2 cells are at (0,7) and (1,7), so this hand is worth keeping while I work on them. I am at (4,7).", "mode": "clean", "goal": {"type": "move_to", "cell": [2, 7], "why": "get within reach of the G2 cells at col 7"}, "note": {"op": "none", "id": 0, "text": ""}, "action": 0}
4.5 Example, changing the goal after an event:
{"why_this_move": "That shot changed nothing at all, so window [3, 1] leaves G0 alone - the third window I have fired at G0 with no effect. I am standing on T2 and have never held [2, 0]. Picking up 2 here starts it.", "mode": "explore", "goal": {"type": "pick_up", "cell": [6, 4], "why": "start an untried window [2, 0]"}, "note": {"op": "add", "id": 0, "text": "every order I test costs a walk to the stations and back, so fire at every shade within reach before going back"}, "action": 8}

5. YOUR NOTES
5.1 What your shots did is kept for you under RULES YOU KNOW, for the whole episode. You never have to write a shot down, and you should not: re-recording one only clutters your notes.
5.2 What is NOT kept for you is how to play. "note" in your reply is one EDIT to your notes, which are shown back to you every step. Every entry is numbered, like [3], and keeps that number for the whole trial. Write what you have worked out and what you mean to do: a way of working that paid off or wasted your time, what the river needed before apples appeared, what the other agents seem to be doing and whether to keep clear of them or follow them, what you will try next and why.
5.3 The four edits. add: a new entry in "text". replace: entry [id] becomes "text" -- use it the moment an entry turns out wrong or you can say it better, and that includes the lessons and plans you carried in from the last trial, which are the entries most likely to be wrong. delete: entry [id] is removed, for an entry that is wrong with nothing to put in its place. none: leave the notes as they are, which is right on most steps. A wrong entry left standing will mislead you on every step until you change it.
5.4 When a trial ends you are asked to rewrite your notes into a short list. THAT LIST IS ALL YOU KEEP of them: the step-by-step lines are dropped, and only what you put in the list is carried into the next trial. RULES YOU KNOW carries over on its own, so spend the list on how to play.
"""


# ---------------------------------------------------------------- the trial-boundary call
# Not a move. The model is shown its notebook and rewrites it, and the answer BECOMES the notebook.
SUMMARISE = """Trial {trial} of {outer} just ended. Before trial {next_trial} starts, rewrite your notes.

How the trial went:
{outcome}

{notebook}

Write down what you have worked out about PLAYING this game. Not which pick-up order is which tool: that is kept for you under RULES YOU KNOW and carries into the next trial by itself, so a line spent on it is a line wasted. Spend them on the rest: an order of doing things that paid off, a way of working that cost you steps, how far the river had to come down before apples appeared, what the other agents did and what you will do about it. Drop anything this trial contradicted. Then list what you still mean to try.

This list is ALL YOU KEEP of your notes. The lines above are dropped, and what you write now is what you will be shown at every step of trial {next_trial}.

Reply with ONE JSON object and nothing else:
{"lessons": ["<something you worked out about how to play>"],
 "todo": ["<something you mean to try next trial>"]}
At most {max_rules} lessons and {max_open} things to try, one short line each."""

SUMMARISE_SCHEMA = {
    "type": "object",
    "properties": {
        "lessons": {"type": "array", "items": {"type": "string", "maxLength": 240}},
        "todo": {"type": "array", "items": {"type": "string", "maxLength": 240}},
    },
    "required": ["lessons", "todo"],
    "additionalProperties": False,
}


AGENT_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# --random-layout: the map is drawn per episode, so nothing in the prompt may say where anything is.
# The Schelling-diagram experiments (2026-09-25) give each agent a ROLE. build_system(role=...) then states 1.1 as the
# scoring rule only (the same words for both roles) and adds 1.5 after 1.4; without a role the prompt is unchanged.
ROLE_SCORE = ("1.1 Score = apples eaten. Only the apples YOU eat count for you.",
              "1.1 Score = apples eaten. Your score counts only the apples you eat yourself.")
ROLE_AFTER = "1.4 You are told only what YOU cleared and ate. You never see other agents' scores, hands or windows."
ROLES = {
    "cooperator": ("1.5 YOUR ROLE: You care about the whole group, not only about yourself. You are willing to give up some of your "
                   "own apples, and to spend your time on work that helps everyone, if it lets the others eat more. The group's "
                   "total matters more to you than your own score."),
    "defector": ("1.5 YOUR ROLE: You care only about your own score. Your goal is to eat as many apples as you can yourself. What "
                 "the other agents eat does not matter to you."),
}


RANDOM_LAYOUT_TEXT = [
    ("1.2 Apples grow only in the orchard, and only while the river is clean enough.",
     "1.2 Apples grow only in the orchard, and only while the river is clean enough. Where the river, the orchard and the stations are "
     "changes from episode to episode and stays the same for all trials of an episode: find them by looking around."),
    ("in a row between the river and the orchard, several of each.", "in a chain between the river and the orchard, several of each."),
    ("row 0 is the NORTH edge (river) and grows southward", "row 0 is the NORTH edge and grows southward"),
    ("T<n> = station of tool n, AP = apple,", "T<n> = station of tool n, AP = apple, ,, = orchard soil (apples grow there),"),
    # 4.4 walks north to the river on the fixed map; on a random one the waste can be in any direction
    ('{"why_this_move": "Both shots I have fired since my window became [2, 0] turned a G2 cell into G1, which nothing I held before did. Two more G2 cells are at (0,7) and (1,7), so this hand is worth keeping while I work on them. I am at (4,7).", "mode": "clean", "goal": {"type": "move_to", "cell": [2, 7], "why": "get within reach of the G2 cells at col 7"}, "note": {"op": "none", "id": 0, "text": ""}, "action": 0}',
     '{"why_this_move": "Both shots I have fired since my window became [2, 0] turned a G2 cell into G1, which nothing I held before did. Another G2 is at (6,3), so this hand is worth keeping while I work on it. I am at (6,8).", "mode": "clean", "goal": {"type": "move_to", "cell": [6, 5], "why": "get within reach of the G2 cells"}, "note": {"op": "none", "id": 0, "text": ""}, "action": 3}'),
]


def build_system(n_agents, inner, outer, me=0, tools="shared", layout="fixed", role=None):
    """me = this agent's index: the prompt names it (A, B, ...) and lists every agent's letter.
    tools = "shared" or "per-colour": an environment switch that no longer reaches the prompt, since 2.3 names no count.
    layout = "fixed" (the mini map, river north and orchard south) or "random" (a map drawn per episode, no geography given).
    role = None (the prompt of every experiment so far) or a key of ROLES (the Schelling-diagram runs)."""
    assert layout in ("fixed", "random"), layout
    text = SYSTEM
    if role is not None:
        assert text.count(ROLE_SCORE[0]) == 1 and text.count(ROLE_AFTER) == 1
        text = text.replace(ROLE_SCORE[0], ROLE_SCORE[1]).replace(ROLE_AFTER, ROLE_AFTER + "\n" + ROLES[role])
    if layout == "random":
        for old, new in RANDOM_LAYOUT_TEXT:
            assert text.count(old) == 1, old[:60]
            text = text.replace(old, new)
    return (text.replace("{n_agents}", str(n_agents))
            .replace("{inner}", str(inner)).replace("{outer}", str(outer))
            .replace("{me}", AGENT_LETTERS[me]).replace("{letters}", ", ".join(AGENT_LETTERS[:n_agents]))
            .replace("{other}", next(L for L in AGENT_LETTERS[:max(n_agents, 2)] if L != AGENT_LETTERS[me])))   # example letter = some agent that is not the reader


def build_summarise(notebook_text, trial, outer, max_rules=12, max_open=4, outcome=""):
    """The question asked at a trial boundary; trial is 0-based, so it names trial+1 as the one that just ended.
    outcome = the trial's own result line, without which the model is judging what paid off blind."""
    return (SUMMARISE.replace("{notebook}", notebook_text)
            .replace("{outcome}", outcome.rstrip() or "(no record of this trial)")
            .replace("{trial}", str(trial + 1))
            .replace("{next_trial}", str(trial + 2)).replace("{outer}", str(outer))
            .replace("{max_rules}", str(max_rules)).replace("{max_open}", str(max_open)))


# ---------------------------------------------------------------- user message
HEADERS = {
    "progress": None,                       # first line, no header
    "rules": "RULES YOU KNOW",              # the harness's own bookkeeping: every order fired with, and what it did
    "notes": "YOUR NOTES (what you worked out about playing)",
    "earlier": "EARLIER TRIALS",
    "social": "OTHER AGENTS (what you have seen yourself)",
    "plan": "YOUR PLAN",
    "last": "LAST STEP",
    "recent": "RECENT STEPS",
    "state": "NOW",
    "actions": "ACTIONS NOW",
}
TRAILER = "Reply with the JSON only."


def compose_user(progress, rules, social, plan, last, recent, state, actions, earlier="", notes=""):
    """The blocks in order. Empty blocks are skipped, their header with them.
    notes = the model's own notebook, rendered by open_cleanup_policy.Notebook."""
    parts = [progress]
    for key, text in (("rules", rules), ("notes", notes), ("earlier", earlier), ("social", social), ("plan", plan),
                      ("last", last), ("recent", recent), ("state", state), ("actions", actions)):
        if text:
            parts.append(f"{HEADERS[key]}\n{text}")
    parts.append(TRAILER)
    return "\n\n".join(parts)
