"""Random maps for open_harvest (layout_pool).

A pool environment must play exactly the game the fixed-map environment plays on the same map, keep its map across
the trials of a meta-episode, draw a new one with the next meta-episode, and lay every reset out on the pool's cells.

Run:
  python -m pytest tests/test_open_harvest_layouts.py -q
"""
import jax
import jax.numpy as jnp
import numpy as onp

from llm_policy.open_harvest_policy import make_env
from opensocialjax.environments.open_harvest.layouts_orig6 import layout_split
from opensocialjax.environments.open_harvest.open_harvest import is_apple

AGENTS, INNER, OUTER = 5, 40, 3


def _rollout(env, key, steps):
    """Random actions from one reset; returns the per-step (grid, reward, layout_id, outer_t)."""
    key, kr = jax.random.split(key)
    obs, state = env.reset(kr)

    def body(carry, _):
        state, key = carry
        key, ka, ks = jax.random.split(key, 3)
        a = jax.random.randint(ka, (AGENTS,), 0, env.num_actions)
        obs, state, r, d, _ = env.step(ks, state, [a[i] for i in range(AGENTS)])
        return (state, key), (obs, state.grid, r, state.layout_id, state.outer_t, d["__all__"])

    return jax.lax.scan(body, (state, key), None, steps)[1]


def test_a_one_map_pool_plays_the_fixed_map_bit_for_bit():
    held = layout_split(on_train=False)
    i = 7
    fixed = make_env(AGENTS, INNER, OUTER, layout=i)
    pooled = _one_map_pool_env(held.subset([i]))   # the same map, as the only entry of a pool
    key = jax.random.PRNGKey(3)
    steps = INNER * OUTER + 5                      # across every trial boundary and into the next meta-episode
    a = jax.jit(lambda k: _rollout(fixed, k, steps))(key)
    b = jax.jit(lambda k: _rollout(pooled, k, steps))(key)
    for name, x, y in zip(("obs", "grid", "reward"), a[:3], b[:3]):
        assert onp.array_equal(onp.asarray(x), onp.asarray(y)), f"{name} differs between the fixed map and its one-map pool"


def _one_map_pool_env(pool):
    """make_env's game with `pool` as the layout pool (make_env itself only takes a whole split)."""
    from opensocialjax.environments.held_out import held_out_harvest_env
    amap = pool.map_ascii(0)
    return held_out_harvest_env(num_agents=AGENTS, num_inner_steps=INNER, num_outer_steps=OUTER, ripen_steps=25,
                                map_ASCII=amap, grid_size=(len(amap), len(amap[0])), frames_till_respawn=12,
                                ripen_range=(15, 20), rot_regrow_range=(2, 5), patch_hues_with_replacement=True,
                                layout_pool=pool)


def test_the_map_is_kept_across_trials_and_redrawn_per_meta_episode():
    env = make_env(AGENTS, INNER, OUTER, layouts="random")
    n = 32
    steps = INNER * OUTER + 1
    out = jax.jit(jax.vmap(lambda k: _rollout(env, k, steps)))(jax.random.split(jax.random.PRNGKey(0), n))
    lid, outer, done = (onp.asarray(x) for x in out[3:])
    T = INNER * OUTER
    assert (lid[:, : T - 1] == lid[:, :1]).all(), "the map changed inside a meta-episode"
    assert set(onp.unique(outer[:, : T - 1])) == set(range(OUTER))
    assert done[:, T - 1].all() and not done[:, : T - 1].any()
    assert (lid[:, T] != lid[:, 0]).mean() > 0.8, "the next meta-episode should draw a new map"


def test_every_map_of_the_split_is_drawn_and_laid_out_on_its_cells():
    for on_train in (False, True):
        env = make_env(AGENTS, INNER, OUTER, on_train=on_train, layouts="random")
        pool = env.layout_pool
        _, st = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(1), 8 * pool.size))
        lid = onp.asarray(st.layout_id)
        assert set(lid.tolist()) == set(range(pool.size)), "some maps of the split are never drawn"
        grid, locs = onp.asarray(st.grid), onp.asarray(st.agent_locs)
        for j in range(0, len(lid), 7):
            trees = pool.arrays["apple"][lid[j]]
            spawns = {tuple(x) for x in pool.arrays["spawns"][lid[j]]}
            assert onp.array_equal(onp.asarray(st.apples[j]), trees)
            on_tree = onp.zeros(grid[j].shape, bool)
            on_tree[trees[:, 0], trees[:, 1]] = True
            assert is_apple(grid[j][on_tree]).all() and not is_apple(grid[j][~on_tree]).any()
            agents = [tuple(x) for x in locs[j][:, :2]]
            assert len(set(agents)) == AGENTS and all(p in spawns for p in agents)
            assert {k: onp.asarray(v).tolist() for k, v in env.layout_arrays(jax.tree.map(lambda x: x[j], st)).items()} == \
                {k: pool.arrays[k][lid[j]].tolist() for k in ("apple", "slot", "spawns")}
