"""The egocentric observation of open_cleanup / open_harvest: every agent sees itself, and ahead is always up.

combine_channels used to read the "self" channel one index too far (agent k saw agent k+1 as itself; JAX clamps
the out-of-range index, so only the last agent was right), and rotate_grid swapped the turns of facings 0 and 2,
turning those two views upside down. For random play in both games, at every step and for every agent:

  * the self channel is on in exactly one cell, (OBS_SIZE - 2, OBS_SIZE // 2), and "other agent" is off there;
  * the inventory shown in that cell is the agent's own;
  * the cell one step ahead in the world is the view cell straight above it, and the cell to its right the view
    cell to its right.

Run:
  python -m pytest tests/test_self_channel.py -q
"""
import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from llm_policy.open_cleanup_policy import make_env as cleanup_env
from llm_policy.open_harvest_policy import make_env as harvest_env
from opensocialjax.environments.open_cleanup import open_cleanup as C
from opensocialjax.environments.open_harvest import open_harvest as H

STEPS = 40
# world (row, col) step of each facing, as STEP in both environments
FORWARD = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def _items_cleanup(code, n_agents):
    one = onp.eye(n_agents + len(C.Items) - 1, dtype=onp.int32)[code - 1] if code > 0 else onp.zeros(n_agents + len(C.Items) - 1, onp.int32)
    item = one[:len(C.Items) - 1]
    return onp.concatenate([item @ onp.asarray(C.KIND_MAP), item @ onp.asarray(C.COLOR_MAP)])


def _items_harvest(code, n_agents):
    one = onp.eye(n_agents + len(H.Items) - 1, dtype=onp.int32)[code - 1] if code > 0 else onp.zeros(n_agents + len(H.Items) - 1, onp.int32)
    return one[:len(H.Items) - 1]


CASES = [
    ("open_cleanup", n, lambda n: cleanup_env(agents=n, inner=400, outer=5, spawn=0.1, skew=1.0, layouts="random",
                                              report_beam=False),
     C.NUM_ITEM_CHANNELS, _items_cleanup) for n in (5, 7)
] + [
    ("open_harvest", n, lambda n: harvest_env(n, 300, 5, respawn_wait=12, rates=(2, 1, 0.5), layouts="random"),
     len(H.Items) - 1, _items_harvest) for n in (5, 7)
]


@pytest.mark.parametrize("name,n,build,self_ch,items_of", CASES, ids=[f"{c[0]}-{c[1]}" for c in CASES])
def test_every_agent_sees_itself_upright(name, n, build, self_ch, items_of):
    env = build(n)
    S = env.OBS_SIZE
    me = (S - 2, S // 2)
    key = jax.random.PRNGKey(0)
    key, kr = jax.random.split(key)
    obs, state = env.reset(kr)
    step = jax.jit(env.step)
    facings = set()
    for t in range(STEPS + 1):
        o, grid = onp.asarray(obs), onp.asarray(state.grid)
        locs, invs = onp.asarray(state.agent_locs), onp.asarray(state.agent_invs)
        rows, cols = grid.shape
        for i in range(n):
            on = onp.argwhere(o[i, ..., self_ch] > 0)
            assert [tuple(p) for p in on] == [me], f"{name} step {t}: agent {i} sees itself at {on.tolist()}, not {me}"
            assert o[i, me[0], me[1], self_ch + 1] == 0, f"{name} step {t}: agent {i} is also 'another agent'"
            assert onp.array_equal(o[i, me[0], me[1], self_ch + 7:self_ch + 9], invs[i]), \
                f"{name} step {t}: agent {i} is shown someone else's inventory"
            r, c, d = (int(v) for v in locs[i])
            facings.add(d)
            fr, fc = FORWARD[d]
            for (vr, vc), (wr, wc) in (((me[0] - 1, me[1]), (r + fr, c + fc)),          # ahead -> up
                                       ((me[0], me[1] + 1), (r + fc, c - fr))):         # right (dc, -dr) -> right
                code = int(grid[wr, wc]) if 0 <= wr < rows and 0 <= wc < cols else 1            # off the map: Items.wall
                assert onp.array_equal(o[i, vr, vc, :self_ch], items_of(code, n)), \
                    f"{name} step {t}: agent {i} facing {d}: view cell {(vr, vc)} is not world cell {(wr, wc)}"
        key, ka, ks = jax.random.split(key, 3)
        a = jax.random.randint(ka, (n,), 0, env.num_actions)
        obs, state, *_ = step(ks, state, [a[j] for j in range(n)])
    assert facings == {0, 1, 2, 3}, f"random play only faced {facings}; the test needs every facing"
