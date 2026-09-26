"""Design gate for open_harvest: scripted policies with / without the hidden rules.

  oracle   knows the pay orders: eats apples at the stage that pays 1 (and over-ripe ones about to rot), but only when the
           emptied cell keeps >= `keep` hanging neighbours (default 3 = full base regrowth rate), so patches stay productive;
           waits by a patch otherwise.  oracle_fast: same but targets the stage whose cell regrows fastest.
  oracle_opt  per colour eats at (or after) the stage s* that maximises pay[s] / (mean regrow time at speed[s] + ripen*s), i.e. the
           steady-state points per step of one tree cell; the true optimum when regrowth does not depend on neighbours (--keep 0)
  greedy   nearest apple, any stage, any hue, no restraint
  patient  nearest apple but only at stage >= 1 (ripe/over), no restraint (knows nothing about peaks)
  leave    never eats (stays put)

usage: python tests/harvest_gate.py [--trials 3] [--steps 500] [--seeds 3] [--keep 3] [--ripen 25]
"""
import argparse, collections, sys
from collections import deque
import jax, jax.numpy as jnp, numpy as onp
sys.path.insert(0, '.')
import opensocialjax
from opensocialjax.environments.open_harvest.open_harvest import (Items, APPLE_BASE, NUM_APPLE_CODES, is_apple, apple_hue, apple_stage, NUM_RIPE_STAGES)
from opensocialjax.environments.discovery_rules import decode_harvest_ruleset, REGROW_RATES, PAY_ORDERS, STAGE_PAY
from opensocialjax.environments.held_out import held_out_harvest_env

MOVES = {(0, 1): 2, (0, -1): 3, (1, 0): 4, (-1, 0): 5}     # (drow, dcol) -> action id; STEP_MOVE: 2 -> col+1, 3 -> col-1, 4 -> row+1, 5 -> row-1
STAY = 6


def bfs_step(grid, start, goals, agents_pos, blocked=frozenset()):
    """First move (action id) on a shortest path from start to any goal cell; None if unreachable / already there.
    ``blocked``: cells never to step on (apples the policy does not want to eat -- walking onto an apple eats it)."""
    H, W = grid.shape
    if start in goals:
        return None
    prev = {start: None}; q = deque([start])
    while q:
        cur = q.popleft()
        if cur in goals:
            # walk back to the first step
            while prev[cur] != start:
                cur = prev[cur]
            return MOVES[(cur[0] - start[0], cur[1] - start[1])]
        for d, a in MOVES.items():
            nxt = (cur[0] + d[0], cur[1] + d[1])
            if not (0 <= nxt[0] < H and 0 <= nxt[1] < W) or nxt in prev:
                continue
            c = int(grid[nxt])
            if c == Items.wall or (c >= len(Items) and nxt in agents_pos and nxt != start) or (nxt in blocked and nxt not in goals):
                continue
            prev[nxt] = cur; q.append(nxt)
    return None


NB12 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]   # the env's regrowth neighbourhood


def nb_hanging(grid, c):
    """Apples hanging in the 12-cell neighbourhood the env counts for regrowth of cell c (c itself excluded)."""
    H, W = grid.shape
    return sum(1 for dr, dc in NB12 if 0 <= c[0] + dr < H and 0 <= c[1] + dc < W and is_apple(grid[c[0] + dr, c[1] + dc]))


def policy_actions(kind, env, state, rates, peaks, keep, hues_of_cell, leave=0.0):
    grid = onp.array(state.grid); locs = onp.array(state.agent_locs); values = onp.array(state.values)
    agents_pos = {tuple(l[:2]) for l in locs}
    acts = []
    cells = [tuple(c) for c in onp.array(env.SPAWNS_APPLE)]
    hanging = collections.Counter(hues_of_cell[c] for c in cells if is_apple(grid[c]))
    size = collections.Counter(hues_of_cell[c] for c in cells)          # cells per colour (= per patch on the mini map)
    for i in range(env.num_agents):
        me = tuple(locs[i][:2])
        if kind == 'leave':
            acts.append(STAY); continue
        apples = [c for c in cells if is_apple(grid[c])]
        if kind == 'greedy':
            goals = set(apples)
        elif kind == 'patient':
            goals = {c for c in apples if apple_stage(grid[c]) >= 1}
        else:   # oracle / oracle_fast: eat at the target stage, but only apples whose cell will still have >= keep hanging
                # neighbours (12-cell neighbourhood) once emptied, so it regrows at the full base rate; over-ripe apples
                # are taken under the same guard (they would rot anyway). Keeps every patch dense enough to keep producing.
            goals = set()
            for c in apples:
                h = hues_of_cell[c]; s = apple_stage(grid[c])
                if values[i, h] == 0:
                    continue
                if kind == 'oracle_opt':
                    if s < peaks[h]:
                        continue                                # not yet at the best stage for this colour
                elif s != peaks[h] and s != NUM_RIPE_STAGES - 1:
                    continue                                    # wait for the target stage (or rescue an apple about to rot)
                if keep > 0 and nb_hanging(grid, c) < keep:
                    continue                                    # leave it: its cell would regrow too slowly
                if leave > 0 and hanging[h] - 1 < leave * size[h]:
                    continue                                    # restraint: keep this share of the patch hanging (it rots and reseeds)
                goals.add(c)
                hanging[h] -= 1                                 # claimed by this agent this step
            if not goals:
                # wait next to the patch that has the most apples close by
                best = None
                for c in apples:
                    h = hues_of_cell[c]
                    if values[i, h] == 0: continue
                    d = abs(c[0] - me[0]) + abs(c[1] - me[1])
                    score = (1 + nb_hanging(grid, c)) / (1 + d)
                    if best is None or score > best[0]: best = (score, c)
                if best is None:
                    acts.append(STAY); continue
                # stand adjacent to it (not on it): goal = neighbours of the apple that are not apples
                c = best[1]; nb = {(c[0] + d[0], c[1] + d[1]) for d in MOVES}
                nb = {n for n in nb if 0 <= n[0] < grid.shape[0] and 0 <= n[1] < grid.shape[1] and int(grid[n]) == Items.empty}
                a = bfs_step(grid, me, nb, agents_pos, blocked=set(apples)); acts.append(STAY if a is None else a); continue
        if not goals:
            acts.append(STAY); continue
        a = bfs_step(grid, me, goals, agents_pos, blocked=set() if kind == 'greedy' else set(apples) - goals)   # do not eat apples on the way
        acts.append(STAY if a is None else a)
        # claim: remove the targeted apple from the shared pool so agents spread out
        if a is not None:
            pass
    return jnp.array(acts)


def run(kind, env, key, trials, steps, keep, rates=None, peaks=None, leave=0.0, recorder=None):
    obs, st = env.reset(key)
    rates_idx, peaks_arr = decode_harvest_ruleset(st.rule_encoding)
    pay = [[STAGE_PAY[r] for r in PAY_ORDERS[int(o)]] for o in onp.array(peaks_arr)]; speed = [[REGROW_RATES[r] for r in PAY_ORDERS[int(o)]] for o in onp.array(rates_idx)]
    rates = speed; peaks = [PAY_ORDERS[int(o)].index(0) for o in onp.array(rates_idx if kind == 'oracle_fast' else peaks_arr)]   # target stage: pays 1 (oracle) / regrows fastest (oracle_fast)
    if kind == 'oracle_opt':   # stage with the best steady-state points per step for one cell: pay / (expected regrow wait + steps to reach the stage)
        base = float(env.REGROW_BASE[0]); ripen = env.ripen_steps
        peaks = [max(range(NUM_RIPE_STAGES), key=lambda s_: pay[h][s_] / (1.0 / (base * speed[h][s_]) + ripen * s_)) for h in range(len(pay))]
    SP = onp.array(env.SPAWNS_APPLE); hues = onp.array(st.cell_hues); hues_of_cell = {tuple(SP[k]): int(hues[k]) for k in range(len(SP))}
    per_trial = []
    score = onp.zeros(env.num_agents); eaten = 0; rotted = 0; prev_grid = onp.array(st.grid)
    for t in range(trials * steps):
        acts = policy_actions(kind, env, st, rates, peaks, keep, hues_of_cell, leave)
        key, k = jax.random.split(key)
        obs, st, r, d, info = env.step_env(k, st, acts)
        g = onp.array(st.grid)
        score += onp.array(info['original_rewards']).reshape(-1) / env.num_agents
        eaten += int(onp.array(info['eaten']).sum())
        # rotted: over-ripe apple cell that became empty without an agent stepping on it
        rot = (apple_stage(prev_grid) == 2) & is_apple(prev_grid) & (g == Items.empty); rotted += int(rot.sum())
        prev_grid = g
        if recorder is not None:
            recorder.set_caption(f"{kind}" + (f"  leave {leave:.0%}" if leave else ""))
            recorder(st, {"trial": t // steps, "t": t % steps, "held": 0, "crafted": 0,
                          "reward": float(onp.array(info['original_rewards']).sum() / env.num_agents), "fired_agents": []}, int(st.inner_t) == 0 and t > 0)
        if int(st.inner_t) == 0 and t > 0:
            cells = g[SP[:, 0], SP[:, 1]] if False else prev_grid[SP[:, 0], SP[:, 1]]
            per_trial.append(dict(score=score.round(1).tolist(), eaten=eaten, rotted=rotted)); score = onp.zeros(env.num_agents); eaten = 0; rotted = 0
    return per_trial, (pay, speed), peaks, onp.array(st.values).tolist()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--trials', type=int, default=3); ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--seeds', type=int, default=3); ap.add_argument('--keep', type=int, default=3, help='oracle: min hanging neighbours an emptied cell must keep'); ap.add_argument('--ripen', type=int, default=25)
    ap.add_argument('--policies', default='oracle,oracle_fast,oracle_opt,greedy,patient,leave')
    ap.add_argument('--rates', default='2,1,0.5', help='regrowth multipliers for speed rank 0/1/2 (2026-09-19: slow 0.5)')
    ap.add_argument('--ripen-range', default='15,20', help='steps per stage drawn per apple (old rules: 25,25)'); ap.add_argument('--rot-range', default='2,5', help='rot countdown drawn per rot (old rules: 5,5)'); ap.add_argument('--pay', default='1,0.5,0.25', help='points for pay rank 0/1/2')
    ap.add_argument('--base', default='0.025,0.005,0.001,0', help='base regrowth probability by hanging neighbours >=3/2/1/0 (vanilla default)')
    ap.add_argument('--leave', type=float, default=0.0, help='oracle restraint: share of each patch that must stay hanging (rots and reseeds)')
    ap.add_argument('--sweep', action='store_true', help='grid over rates x unripe x ripen, oracle vs greedy means')
    ap.add_argument('--map', choices=['mini', 'full'], default='mini')
    ap.add_argument('--harness-seed', type=int, default=None, help='use the exact reset key of llm_policy/open_harvest_policy.py --seed N (episode 0)')
    ap.add_argument('--gif-dir', default=None, help='write <gif-dir>/<policy>_trial<k>.gif of the first trial (every step)')
    ap.add_argument('--gif-policies', default='oracle,greedy', help='which policies to record when --gif-dir is set')
    a = ap.parse_args()
    def make_rec(kind, env):
        if not a.gif_dir or kind not in a.gif_policies.split(','):
            return None
        from llm_policy.record import GifRecorder
        return GifRecorder(env, a.gif_dir, kind, trials=[0], every=1, fps=8)
    def make(rates, pay, ripen, base=None):
        extra = {}
        if a.map == 'mini':
            from llm_policy.open_harvest_policy import MINI_MAP
            extra = {"map_ASCII": MINI_MAP, "grid_size": (len(MINI_MAP), max(len(r) for r in MINI_MAP))}
        return held_out_harvest_env(num_agents=3, num_inner_steps=a.steps, num_outer_steps=a.trials, ripen_steps=ripen,
                                    regrow_rates=tuple(float(x) for x in rates.split(',')), stage_pay=tuple(float(x) for x in pay.split(',')),
                                    regrow_base=tuple(float(x) for x in (base or a.base).split(',')),
                                    ripen_range=tuple(int(x) for x in a.ripen_range.split(',')), rot_regrow_range=tuple(int(x) for x in a.rot_range.split(',')), **extra)
    if a.sweep:
        import io, contextlib
        print(f"{'rates':>14} {'pay':>12} {'ripen':>5} | {'oracle':>7} {'greedy':>7} {'patient':>7} | oracle/greedy   (team score per trial, mean over seeds x trials)")
        for rates in ('2,1,0.25', '4,1,0.25'):
            for pay in ('1,0.5,0.25', '1,0.5,0', '1,0.25,0.25'):
                for ripen in (15, 25):
                    with contextlib.redirect_stdout(io.StringIO()):
                        env = make(rates, pay, ripen)
                    res = {}
                    for kind in ('oracle', 'greedy', 'patient'):
                        tots = []
                        for seed in range(a.seeds):
                            per, *_ = run(kind, env, jax.random.PRNGKey(100 + seed, recorder=make_rec(kind, env)), a.trials, a.steps, a.keep)
                            tots += [sum(p['score']) for p in per]
                        res[kind] = sum(tots) / len(tots)
                    print(f"{rates:>14} {pay:>12} {ripen:>5} | {res['oracle']:>7.1f} {res['greedy']:>7.1f} {res['patient']:>7.1f} | {res['oracle']/max(1e-6,res['greedy']):.2f}", flush=True)
        raise SystemExit
    env = make(a.rates, a.pay, a.ripen)
    for seed in range(1 if a.harness_seed is not None else a.seeds):
        key = jax.random.PRNGKey(100 + seed)
        if a.harness_seed is not None:            # same reset key as open_harvest_policy.run_episode: split(PRNGKey(seed), 3)[1]
            _, key, _ = jax.random.split(jax.random.PRNGKey(a.harness_seed), 3)
        print(f"== seed {a.harness_seed if a.harness_seed is not None else seed}")
        for kind in a.policies.split(','):
            per, (pay, speed), peaks, values = run(kind, env, key, a.trials, a.steps, a.keep, leave=a.leave, recorder=make_rec(kind, env))
            tot = [round(sum(p['score']), 1) for p in per]
            print(f"  {kind:11s} team score/trial {tot} | eaten {[p['eaten'] for p in per]} | rotted {[p['rotted'] for p in per]}"
                  + (f"   [pay unripe/ripe/over {pay} speed {speed}]" if kind == 'oracle' else ''))
