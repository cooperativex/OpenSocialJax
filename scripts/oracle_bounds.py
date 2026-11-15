"""Scripted reference policies on the LLM experiments' exact problems: seeds 906-908 of open_cleanup_5ag / open_harvest_5ag,
the same map, hidden rule and colours (the LLM run's reset key) and the same step-key chain, 5 agents, 5 trials. Every
policy reads the true rule and the map from the state (full information), so the numbers bound the map, not the learning.

  selfish  every agent maximises only what it eats right now and never pays for others.
           Cleanup: nobody cleans the river (cleaning feeds everyone, and the river opens fully polluted) -> 0 apples.
           Harvest: knows the rule; walks to the hanging apple with the most points per step of walking, at any stage,
           never leaves any to regrow (no zapping).
  coop     the team's best scripted plan we found: a parameter sweep, best total per seed. A lower bound on the true optimum
           (a joint plan of 5 agents over 2000 / 1500 steps is not computable), not the optimum itself.
           Cleanup: k cleaners hold the working tools (crafted from the true recipes at the nearest stations) and fire
           where their beam acts on the most waste, the others eat; pipeline (one tool class per cleaner, by workload
           light:mid:dark = 3:2:1) or flexible (each cleaner takes the class with the most waste); eaters clean while
           no apple hangs (help); cleaners eat while the river is below a pollution share (theta).
           Harvest: tests/harvest_gate.py's oracles (eat at the best-paying stage / the best points per step / the
           fastest-regrowing stage, leave each emptied cell `keep` hanging neighbours, keep `leave` of each colour
           hanging), with and without territories (each agent works its own patches only).

usage: python scripts/oracle_bounds.py cleanup|harvest [--seeds 906 907 908] [--quick] [--out DIR]
"""
import argparse, collections, itertools, json, os, sys, time
from collections import deque

import jax
import numpy as onp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_AGENTS, OUTER = 5, 5
DIRS4 = [(-1, 0), (1, 0), (0, 1), (0, -1)]


def bfs(passable, start):
    """Shortest walking distances from start and the predecessor of every reached cell."""
    dist = {start: 0}; prev = {start: None}; q = deque([start])
    while q:
        cur = q.popleft()
        for dr, dc in DIRS4:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt not in dist and passable(nxt):
                dist[nxt] = dist[cur] + 1; prev[nxt] = cur; q.append(nxt)
    return dist, prev


def first_step(prev, start, goal):
    cur = goal
    while prev[cur] != start:
        cur = prev[cur]
    return (cur[0] - start[0], cur[1] - start[1])


# ================================================================ Cleanup
def cleanup_problem_env(seed):
    from llm_policy import open_cleanup_policy as C
    cfg = json.load(open(os.path.join(REPO, "experiments/open_cleanup_5ag/config.json")))["setting"]
    env = C.make_env(cfg["agents"], cfg["inner"], cfg["outer"], cfg["spawn"], cfg["skew"], tools=cfg["tools"], layouts="random")
    return env, cfg


class CleanupTeam:
    """One scripted team. roles: per agent "eat" or a cleaner role (light / mid / dark / darkmid / flex)."""
    MOVE = {(-1, 0): 0, (1, 0): 1, (0, 1): 2, (0, -1): 3}           # raw action ids: north, south, east, west

    def __init__(self, env, state, roles, help_=False, theta=0.0):
        from llm_policy import open_cleanup_policy as C
        from opensocialjax.environments.open_cleanup.open_cleanup import DIRT_BASE, NUM_DIRT_TYPES, Items
        self.C, self.DIRT_BASE, self.NDT, self.Items = C, DIRT_BASE, NUM_DIRT_TYPES, Items
        self.env, self.roles, self.help, self.theta = env, list(roles), help_, theta
        self.recipes = C.true_recipes(env, state)
        self.dom = int(state.dom_type)
        self.cls = {"light": ("hue", self.dom), "mid": "mid", "dark": "dark"}
        lay = env.layout_arrays(state)
        self.waste_cells = [tuple(int(x) for x in rc) for rc in lay["waste_cells"]]
        self.denominator = len(self.waste_cells) + len(lay["river"])       # the env's dirtFraction denominator
        self.orchard = [tuple(int(x) for x in rc) for rc in lay["apple"]]
        tg = lay["tool_grid"]
        self.stations = collections.defaultdict(list)
        for rc in lay["tools"]:
            self.stations[int(tg[rc[0], rc[1]])].append((int(rc[0]), int(rc[1])))
        self.H, self.W = onp.asarray(state.grid).shape
        self.blocked = set()                                            # cells a move into failed for no visible reason
        self.last = {}                                                  # agent -> (pos, move delta) of its last move
        oc = onp.array(self.orchard).mean(0)
        self.wait_cells = sorted(self.orchard, key=lambda c: abs(c[0] - oc[0]) + abs(c[1] - oc[1]))

    # ---- helpers
    def passable(self, grid, others):
        W = int(self.Items.wall)
        return lambda c: (0 <= c[0] < self.H and 0 <= c[1] < self.W and int(grid[c]) != W and c not in others and c not in self.blocked)

    def beam(self, pos, d):
        return [c for c in self.C.beam_cells(self.env, (pos[0], pos[1], d))]

    def hits(self, cls, code):
        return self.C.effect_of(cls, code) is not None

    def go(self, me, prev, target):
        return self.MOVE[first_step(prev, me, target)]

    # ---- one step for every agent
    def act(self, state):
        grid = onp.asarray(state.grid); locs = onp.asarray(state.agent_locs); picks = onp.asarray(state.pickups)
        pos = [(int(l[0]), int(l[1])) for l in locs]; face = [int(l[2]) for l in locs]
        for i, (p0, dlt) in list(self.last.items()):                   # a move that did not happen: learn the cell
            if pos[i] == p0:
                tgt = (p0[0] + dlt[0], p0[1] + dlt[1])
                if tgt not in pos:
                    self.blocked.add(tgt)
        self.last = {}
        waste = {c: int(grid[c]) - self.DIRT_BASE for c in self.waste_cells if self.DIRT_BASE <= int(grid[c]) < self.DIRT_BASE + self.NDT}
        frac = len(waste) / self.denominator
        apples = [c for c in self.orchard if int(grid[c]) == int(self.Items.apple)]
        claimed_waste, claimed_apple = set(), set()
        acts = [7] * N_AGENTS
        order = sorted(range(N_AGENTS), key=lambda i: self.roles[i] in ("eat", "defect"))   # cleaners claim first
        for i in order:
            me = pos[i]; others = set(pos[:i] + pos[i + 1:])
            dist, prev = bfs(self.passable(grid, others), me)
            role = self.roles[i]
            if role == "defect":                                        # selfish: eats only, takes the nearest apple, no claims
                acts[i] = self.eat(i, me, dist, prev, apples, set(), claim=False)
                continue
            if role == "eat":
                if apples or not self.help:
                    acts[i] = self.eat(i, me, dist, prev, apples, claimed_apple)
                    continue
                role = "flex"                                           # no apple hangs anywhere: help clean
            elif apples and frac <= self.theta:
                acts[i] = self.eat(i, me, dist, prev, apples, claimed_apple)
                continue
            a = self.clean(i, role, me, face[i], [int(x) for x in picks[i] if int(x) >= 0], dist, prev, waste, claimed_waste)
            if a is None:                                                # nothing this cleaner's tools can act on
                a = self.eat(i, me, dist, prev, apples, claimed_apple) if apples else 7
            acts[i] = a
        for i, a in enumerate(acts):
            if a in (0, 1, 2, 3):
                self.last[i] = (pos[i], {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}[a])
        return acts, frac

    def eat(self, i, me, dist, prev, apples, claimed, claim=True):
        cand = [c for c in apples if c in dist and c not in claimed]
        if cand:
            t = min(cand, key=lambda c: (dist[c], c))
            if claim:
                claimed.add(t)
            return self.go(me, prev, t) if t != me else 7
        spots = [c for c in self.wait_cells if c in dist]
        if not spots:
            return 7
        t = spots[min(i, len(spots) - 1)]
        return self.go(me, prev, t) if t != me else 7

    def clean(self, i, role, me, d, picks, dist, prev, waste, claimed):
        count = {k: sum(self.hits(self.cls[k], code) for c, code in waste.items() if c not in claimed) for k in self.cls}
        held = next((k for k in self.cls if tuple(picks[-2:]) == tuple(self.recipes[self.cls[k]])), None)
        allowed = {"light": ["light"], "mid": ["mid"], "dark": ["dark"], "darkmid": ["mid", "dark"], "flex": ["light", "mid", "dark"]}[role]
        if held in allowed and count[held] > 0:
            k = held
        else:
            live = [k for k in allowed if count[k] > 0]
            if not live:
                return None
            k = max(live, key=lambda k: (count[k], k == "light"))
        cls = self.cls[k]
        if held != k:                                                    # craft it: next pick-up of its recipe
            a, b = self.recipes[cls]
            need = b if (picks and picks[-1] == a) else a
            here = self.stations_at(me)
            if here == need:
                return 8
            cands = [s for s in self.stations[need] if s in dist]
            if not cands:
                return None
            if need == a:                                                # pick the first station with the second in mind
                tgt = min(cands, key=lambda s: dist[s] + min(abs(s[0] - t[0]) + abs(s[1] - t[1]) for t in self.stations[b]))
            else:
                tgt = min(cands, key=lambda s: dist[s])
            return self.go(me, prev, tgt)
        # holding the tool: fire if the beam acts on something now, else go where it acts on the most per step
        now = [c for c in self.beam(me, d) if c in waste and c not in claimed and self.hits(cls, waste[c])]
        if now:
            claimed.update(now); return 6
        best = None
        for p, dp in dist.items():
            for nd in range(4):
                cov = [c for c in self.beam(p, nd) if c in waste and c not in claimed and self.hits(cls, waste[c])]
                if not cov:
                    continue
                turns = (nd - d) % 4; turns = min(turns, 4 - turns)
                score = len(cov) / (dp + turns + 1)
                if best is None or score > best[0]:
                    best = (score, p, nd, cov)
        if best is None:
            return None
        _, p, nd, cov = best
        claimed.update(cov)
        if p != me:
            return self.go(me, prev, p)
        return 4 if (nd - d) % 4 in (1, 2) else 5                        # turn left = facing + 1

    def stations_at(self, me):
        for t, cells in self.stations.items():
            if me in cells:
                return t
        return None


PIPELINE = {1: ["flex"], 2: ["light", "darkmid"], 3: ["light", "mid", "dark"], 4: ["light", "light", "mid", "dark"],
            5: ["light", "light", "mid", "mid", "dark"]}


def cleanup_configs(quick=False):
    yield dict(name="selfish", k=0, mode="-", help=False, theta=0.0)
    ks = (2, 3) if quick else (1, 2, 3, 4, 5)
    for k in ks:
        for mode in (("pipeline",) if k == 1 else ("pipeline", "flex")):
            for help_ in (False, True):
                for theta in ((0.0,) if quick else (0.0, 0.1, 0.2)):
                    if k == 5 and help_:
                        continue                                          # no eaters to help
                    yield dict(name="coop", k=k, mode=mode, help=help_, theta=theta)


def cleanup_run(env, seed, cfg_, seats=None):
    """seats: per agent "C" / "D" (Schelling runs): the cooperators play cfg_'s plan among themselves (cleaners at most
    cfg_["k"], picked nearest the stations), the defectors play selfish. None = every agent plays cfg_."""
    from llm_policy import open_cleanup_policy as C
    key = jax.random.PRNGKey(seed); key, kr, ks = jax.random.split(key, 3)       # run_episode's keys for this run key
    _, state = env.reset(kr)
    lay = env.layout_arrays(state); tg = lay["tool_grid"]
    st_cells = [tuple(int(x) for x in rc) for rc in lay["tools"]]
    locs = onp.asarray(state.agent_locs)
    seats = seats or ["C"] * N_AGENTS
    near = sorted([i for i in range(N_AGENTS) if seats[i] == "C"],
                  key=lambda i: min(abs(int(locs[i][0]) - s[0]) + abs(int(locs[i][1]) - s[1]) for s in st_cells))
    roles = ["eat" if seats[i] == "C" else "defect" for i in range(N_AGENTS)]
    kc = min(cfg_["k"], len(near))
    clean_roles = (PIPELINE[kc] if cfg_["mode"] == "pipeline" else ["flex"] * kc) if kc else []
    for j, r in enumerate(clean_roles):
        roles[near[j]] = r
    team = CleanupTeam(env, state, roles, cfg_["help"], cfg_["theta"])
    step = jax.jit(env.step)
    apples = [0.0] * OUTER; fr = [[] for _ in range(OUTER)]; first = [None] * OUTER; per = [[0.0] * N_AGENTS for _ in range(OUTER)]
    done = False
    while not done:
        trial, t = int(state.outer_t), int(state.inner_t)
        acts, frac = team.act(state)
        fr[trial].append(frac)
        ks, k1 = jax.random.split(ks)
        _, state, rew, dones, info = step(k1, state, [C.RAW_TO_ENV[a] for a in acts])
        own = onp.asarray(info["original_rewards"]).reshape(-1) / N_AGENTS
        got = float(own.sum()); apples[trial] += got
        for i in range(N_AGENTS):
            per[trial][i] += float(own[i])
        if got > 0 and first[trial] is None:
            first[trial] = t
        done = bool(dones["__all__"])
    return dict(team_apples=[round(a, 1) for a in apples], total=round(sum(apples), 1), roles=roles, seats=seats,
                per_agent=[[round(v, 2) for v in t] for t in per],
                mean_pollution=[round(float(onp.mean(f)), 3) for f in fr],
                steps_below_0_4=[int(sum(x < 0.4 for x in f)) for f in fr], first_apple_step=first)


# ================================================================ Harvest
def harvest_problem_env(seed):
    from llm_policy import open_harvest_policy as H
    from opensocialjax.environments.open_harvest.layouts_orig6 import pick_layout
    cfg = json.load(open(os.path.join(REPO, "experiments/open_harvest_5ag/config.json")))["setting"]
    ints = lambda x: [int(v) for v in str(x).split(",")]
    env = H.make_env(cfg["agents"], cfg["inner"], cfg["outer"], cfg["ripen"], False, pick_layout(seed, False), cfg["respawn_wait"],
                     ripen_range=ints(cfg["ripen_range"]), rot_range=ints(cfg["rot_range"]), rates=[float(v) for v in str(cfg["rates"]).split(",")])
    return env, cfg


def patches(cells):
    """Connected groups of tree cells (8-neighbourhood): the map's patches."""
    left = set(cells); out = []
    while left:
        s = left.pop(); comp = [s]; q = deque([s])
        while q:
            r, c = q.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    n = (r + dr, c + dc)
                    if n in left:
                        left.remove(n); comp.append(n); q.append(n)
        out.append(comp)
    return out


def harvest_run(env, seed, cfg_, seats=None):
    """seats: per agent "C" / "D" (Schelling runs): cooperators play cfg_, defectors play selfish. None = all play cfg_."""
    sys.path.insert(0, os.path.join(REPO, "tests"))
    import harvest_gate as G
    from opensocialjax.environments.open_harvest.open_harvest import Items, is_apple, apple_stage, NUM_RIPE_STAGES
    from opensocialjax.environments.discovery_rules import decode_harvest_ruleset, REGROW_RATES, PAY_ORDERS, STAGE_PAY
    key = jax.random.PRNGKey(seed); key, kr, ks = jax.random.split(key, 3)       # the harness run_episode's keys
    _, st = env.reset(kr)
    rates_idx, pay_idx = decode_harvest_ruleset(st.rule_encoding)
    pay = [[STAGE_PAY[r] for r in PAY_ORDERS[int(o)]] for o in onp.array(pay_idx)]
    speed = [[REGROW_RATES[r] for r in PAY_ORDERS[int(o)]] for o in onp.array(rates_idx)]
    kind = cfg_["kind"]
    kinds = [kind if (seats is None or seats[i] == "C") else "selfish" for i in range(N_AGENTS)]
    if kind == "oracle_opt":
        base = float(env.REGROW_BASE[0]); ripen = env.ripen_steps
        target = [max(range(NUM_RIPE_STAGES), key=lambda s_: pay[h][s_] / (1.0 / (base * speed[h][s_]) + ripen * s_)) for h in range(len(pay))]
    else:
        target = [PAY_ORDERS[int(o)].index(0) for o in onp.array(rates_idx if kind == "oracle_fast" else pay_idx)]
    SP = [tuple(int(x) for x in rc) for rc in onp.array(env.SPAWNS_APPLE)]
    hues = onp.array(st.cell_hues); hue_of = {SP[k]: int(hues[k]) for k in range(len(SP))}
    size = collections.Counter(hue_of.values())
    terr = None
    if cfg_.get("territory"):
        groups = sorted(patches(SP), key=len, reverse=True); load = [0] * N_AGENTS; terr = [set() for _ in range(N_AGENTS)]
        locs0 = onp.asarray(st.agent_locs)
        for g in groups:                                                  # biggest patch first, to the least-loaded agent, nearest breaks ties
            cy, cx = onp.mean([c[0] for c in g]), onp.mean([c[1] for c in g])
            i = min(range(N_AGENTS), key=lambda j: (load[j], abs(int(locs0[j][0]) - cy) + abs(int(locs0[j][1]) - cx)))
            terr[i] |= set(g); load[i] += len(g)
    H_, W_ = onp.asarray(st.grid).shape
    MOVE = G.MOVES; STAY = G.STAY
    pts = [0.0] * OUTER; eaten = [0] * OUTER; per = [[0.0] * N_AGENTS for _ in range(OUTER)]
    for _ in range(OUTER * env.num_inner_steps):
        trial = int(st.outer_t)
        grid = onp.asarray(st.grid); locs = onp.asarray(st.agent_locs); values = onp.asarray(st.values)
        pos = [(int(l[0]), int(l[1])) for l in locs]
        apples = [c for c in SP if is_apple(grid[c])]
        hanging = collections.Counter(hue_of[c] for c in apples)
        acts = []
        for i in range(N_AGENTS):
            me = pos[i]; others = set(pos[:i] + pos[i + 1:])
            mine = [c for c in apples if terr is None or c in terr[i]]
            if kinds[i] == "selfish":
                goals = set(apples); avoid = set()
            else:
                goals = set()
                for c in mine:
                    h = hue_of[c]; s = int(apple_stage(grid[c]))
                    if values[i, h] == 0:
                        continue
                    if kind == "oracle_opt":
                        if s < target[h]:
                            continue
                    elif s != target[h] and s != NUM_RIPE_STAGES - 1:
                        continue
                    if cfg_["keep"] > 0 and G.nb_hanging(grid, c) < cfg_["keep"]:
                        continue
                    if cfg_["leave"] > 0 and hanging[h] - 1 < cfg_["leave"] * size[h]:
                        continue
                    goals.add(c); hanging[h] -= 1
                avoid = set(apples) - goals                              # walking onto an apple eats it
            # the env lets an agent onto an empty cell or an apple (which it then eats); anything else blocks
            passable = lambda c: (0 <= c[0] < H_ and 0 <= c[1] < W_ and c not in others and (c in goals or c not in avoid)
                                  and (int(grid[c]) == int(Items.empty) or bool(is_apple(int(grid[c]))) or c == me))
            dist, prev = bfs(passable, me)
            reach = [c for c in goals if c in dist and c != me]
            if reach:
                if kinds[i] == "selfish":                                # most points per step of walking, right now
                    t = max(reach, key=lambda c: (pay[hue_of[c]][int(apple_stage(grid[c]))] / (dist[c] + 1), -dist[c]))
                else:
                    t = min(reach, key=lambda c: dist[c])
                acts.append(MOVE[first_step(prev, me, t)]); continue
            # nothing to eat now: stand next to the own patch with the most hanging apples close by
            pool = mine if mine else ([c for c in SP if c in terr[i]] if terr is not None else apples)
            if not pool:
                acts.append(STAY); continue
            best = min(pool, key=lambda c: (abs(c[0] - me[0]) + abs(c[1] - me[1])) / (1 + G.nb_hanging(grid, c)))
            nb = [(best[0] + dr, best[1] + dc) for dr, dc in DIRS4]
            nb = [n for n in nb if n in dist and int(grid[n]) == int(Items.empty)]
            if not nb or me in nb:
                acts.append(STAY); continue
            t = min(nb, key=lambda n: dist[n]); acts.append(MOVE[first_step(prev, me, t)])
        ks, k1 = jax.random.split(ks)
        _, st, r, d, info = env.step_env(k1, st, jax.numpy.array(acts))
        own = onp.asarray(info["original_rewards"]).reshape(-1) / N_AGENTS
        pts[trial] += float(own.sum())
        for i in range(N_AGENTS):
            per[trial][i] += float(own[i])
        eaten[trial] += int(onp.asarray(info["eaten"]).sum())
    return dict(team_points=[round(p, 2) for p in pts], total=round(sum(pts), 2), eaten=eaten, seats=seats or ["C"] * N_AGENTS,
                per_agent=[[round(v, 2) for v in t] for t in per],
                points_per_apple=round(sum(pts) / max(sum(eaten), 1), 3))


def harvest_configs(quick=False):
    yield dict(name="selfish", kind="selfish", keep=0, leave=0.0, territory=False)
    yield dict(name="greedy_blind", kind="greedy", keep=0, leave=0.0, territory=False)
    for kind in ("oracle",) if quick else ("oracle", "oracle_opt", "oracle_fast"):
        for keep in ((3,) if quick else (0, 1, 2, 3, 4)):
            for leave in ((0.0,) if quick else (0.0, 0.2)):
                for terr in (False, True):
                    yield dict(name="coop", kind=kind, keep=keep, leave=leave, territory=terr)


def harvest_run_any(env, seed, c):
    if c["kind"] == "greedy":                                            # tests/harvest_gate.py's rule-blind greedy, as is
        sys.path.insert(0, os.path.join(REPO, "tests"))
        import harvest_gate as G
        _, kr, _ = jax.random.split(jax.random.PRNGKey(seed), 3)
        per, _, _, _ = G.run("greedy", env, kr, OUTER, env.num_inner_steps, 0)
        tp = [round(sum(p["score"]), 2) for p in per]
        return dict(team_points=tp, total=round(sum(tp), 2), eaten=[p["eaten"] for p in per])
    return harvest_run(env, seed, c)


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("game", choices=["cleanup", "harvest"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[906, 907, 908])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(REPO, f"experiments/open_{a.game}_5ag/bounds")
    os.makedirs(out, exist_ok=True)
    for seed in a.seeds:
        env, _ = (cleanup_problem_env if a.game == "cleanup" else harvest_problem_env)(seed)
        rows = []
        for c in (cleanup_configs if a.game == "cleanup" else harvest_configs)(a.quick):
            t0 = time.time()
            r = (cleanup_run if a.game == "cleanup" else harvest_run_any)(env, seed, c)
            r.update(config=c, seconds=round(time.time() - t0, 1)); rows.append(r)
            print(seed, json.dumps(c), "->", r.get("team_apples", r.get("team_points")), "total", r["total"], f"({r['seconds']} s)", flush=True)
        json.dump(rows, open(os.path.join(out, f"oracle_seed{seed}{'_quick' if a.quick else ''}.json"), "w"), indent=1)
