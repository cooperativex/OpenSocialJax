"""Rendering checks for both environments.

Covers: render output type/shape stability, PIL compatibility, the "agents
never vanish from the grid" invariant (beams must not overwrite agent
stamps), warm-cache render speed, the river background under an agent in
OpenCleanup, and tile_cache instance isolation.

Run:
  python -m pytest tests/test_render.py -q
  python tests/test_render.py          # the same checks, no pytest
"""

import sys
import time

import jax
import numpy as onp
import pytest
from PIL import Image

import opensocialjax
from opensocialjax.registration import REGISTERED_ENVS

STEPS = 30
WARM_TIMING_REPS = 5
# generous bound for a loaded machine; a warm render takes a few ms per frame
MAX_WARM_SECONDS = 0.5


def env_module(env):
    return sys.modules[env.__class__.__module__]


def check_frame(env_id, step, img, expected_shape):
    assert isinstance(img, onp.ndarray), f"{env_id}: render returned {type(img)}"
    assert img.dtype == onp.uint8, f"{env_id}: dtype {img.dtype}"
    assert img.ndim == 3 and img.shape[2] == 3, f"{env_id}: shape {img.shape}"
    if expected_shape is not None:
        assert img.shape == expected_shape, (
            f"{env_id} step {step}: shape changed {expected_shape} -> {img.shape}")
    Image.fromarray(img)  # must not raise
    return img.shape


def run_env(env_id):
    env = opensocialjax.make(env_id)
    mod = env_module(env)
    Actions = mod.Actions
    # force beams so the agent-visibility invariant is exercised
    zap_actions = [a.value for a in Actions if "zap" in a.name]
    agent_codes = onp.asarray(env._agents)

    key = jax.random.PRNGKey(0)
    key, kr = jax.random.split(key)
    _, state = env.reset(kr)
    step_fn = jax.jit(env.step)

    shape = check_frame(env_id, -1, env.render(state), None)

    for t in range(STEPS):
        key, ka, ks = jax.random.split(key, 3)
        if t % 2 == 1:
            actions = [zap_actions[t // 2 % len(zap_actions)]] * env.num_agents
        else:
            actions = [env.action_space(a).sample(jax.random.fold_in(ka, i))
                       for i, a in enumerate(env.agents)]
        _, state, _, _, _ = step_fn(ks, state, actions)

        img = env.render(state)
        check_frame(env_id, t, img, shape)

        grid = onp.asarray(state.grid)
        for code in agent_codes:
            assert (grid == code).any(), (
                f"{env_id} step {t}: agent code {code} missing from grid "
                f"(agent vanished from rendering)")

    # warm-cache timing
    t0 = time.perf_counter()
    for _ in range(WARM_TIMING_REPS):
        env.render(state)
    warm = (time.perf_counter() - t0) / WARM_TIMING_REPS
    assert warm < MAX_WARM_SECONDS, f"{env_id}: warm render {warm:.3f}s/frame"
    print(f"ok: {env_id} ({env.num_agents} agents, warm render {warm*1000:.1f} ms/frame)")


@pytest.mark.parametrize("env_id", REGISTERED_ENVS)
def test_render_rollout(env_id):
    run_env(env_id)


def test_cleanup_river_background():
    """An agent standing in the river must be drawn on water, not on the
    default background (regression: the agent grid stamp destroys the terrain
    value, so render must recover it from the layout)."""
    from opensocialjax.environments.open_cleanup.open_cleanup import Items

    env = opensocialjax.make("open_cleanup")
    _, state = env.reset(jax.random.PRNGKey(0))

    # Move agent 0 onto a river cell and restamp the grid.
    r, c = sorted(env.layout_arrays(state)["river_set"])[0]
    old = onp.asarray(state.agent_locs)[0]
    code = int(onp.asarray(env._agents)[0])
    grid = state.grid.at[int(old[0]), int(old[1])].set(0).at[r, c].set(code)
    locs = state.agent_locs.at[0, 0].set(r).at[0, 1].set(c)
    state = state.replace(grid=grid, agent_locs=locs)

    img = env.render(state)
    d = int(onp.asarray(locs)[0, 2])
    # tile (r, c) after pad/crop (the image has the map's orientation)
    region = img[(r + 1) * 32:(r + 2) * 32,
                 (c + 1) * 32:(c + 2) * 32]

    def tile(terrain, highlight):
        # blit truncates float (highlighted) tiles to uint8; mirror that
        t = env.render_tile(code, agent_dir=d, agent_hat=False,
                            highlight=highlight, tile_size=32, terrain=terrain)
        return t.astype(onp.uint8)

    on_river = [hl for hl in (True, False)
                if onp.array_equal(region, tile(int(Items.river), hl))]
    assert on_river, "agent-in-river tile does not use the river background"
    assert not onp.array_equal(region, tile(None, on_river[0])), \
        "agent-in-river tile still matches the default background"
    print("ok: cleanup river background under agent")


def test_tile_cache_isolation():
    from opensocialjax.environments.open_harvest.open_harvest import OpenHarvest
    e1 = OpenHarvest(num_agents=5)
    e2 = OpenHarvest(num_agents=3)
    assert e1.tile_cache is not e2.tile_cache, "tile_cache shared across instances"
    key = jax.random.PRNGKey(0)
    _, s1 = e1.reset(key)
    _, s2 = e2.reset(key)
    e1.render(s1)
    e2.render(s2)
    assert e1.PLAYER_COLOURS != e2.PLAYER_COLOURS
    print("ok: tile_cache instance isolation")


if __name__ == "__main__":
    failures = []
    for env_id in REGISTERED_ENVS:
        try:
            run_env(env_id)
        except AssertionError as e:
            failures.append(f"{env_id}: {e}")
            print(f"FAIL: {env_id}: {e}")
    for extra in (test_cleanup_river_background, test_tile_cache_isolation):
        try:
            extra()
        except AssertionError as e:
            failures.append(f"{extra.__name__}: {e}")
            print(f"FAIL: {extra.__name__}: {e}")
    if failures:
        sys.exit(f"{len(failures)} rendering check(s) failed")
    print("ALL RENDER TESTS PASSED")
