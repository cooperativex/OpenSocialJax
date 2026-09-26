"""Standalone rendering checks for all environments (no pytest).

Covers: render output type/shape stability, PIL compatibility, the
territory claimed-resource color regression (render used to crash once any
resource was claimed), the "agents never vanish from the grid" invariant
(beams must not overwrite agent stamps), warm-cache render speed, and
tile_cache instance isolation.

Run:
  ulimit -c 0; ulimit -u $(ulimit -Hu)
  export PYTHONPATH=$PWD:$PYTHONPATH
  JAX_PLATFORMS=cpu python tests/test_render.py
"""

import sys
import time

import jax
import numpy as onp
from PIL import Image

import opensocialjax
from opensocialjax.registration import REGISTERED_ENVS

STEPS = 30
WARM_TIMING_REPS = 5
# generous bound for a loaded login node; the old render was 0.14-0.34 s/f
MAX_WARM_SECONDS = 0.15


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
    Actions = getattr(mod, "Actions", None)
    Items = getattr(mod, "Items", None)

    zap_actions = []
    if Actions is not None:
        zap_actions = [a.value for a in Actions
                       if "zap" in a.name or "claim" in a.name or "clean" in a.name]

    has_grid_stamps = Items is not None and hasattr(env, "_agents") \
        and hasattr(env.reset(jax.random.PRNGKey(0))[1], "grid") \
        and env_id not in ("coop_mining", "lb_foraging")
    agent_codes = onp.asarray(env._agents) if has_grid_stamps else None

    key = jax.random.PRNGKey(0)
    key, kr = jax.random.split(key)
    _, state = env.reset(kr)
    step_fn = jax.jit(env.step)

    shape = check_frame(env_id, -1, env.render(state), None)
    saw_claim = False

    for t in range(STEPS):
        key, ka, ks = jax.random.split(key, 3)
        if zap_actions and t % 2 == 1:
            # force beams so the agent-visibility invariant is exercised
            actions = [zap_actions[t // 2 % len(zap_actions)]] * env.num_agents
        else:
            actions = [env.action_space(a).sample(jax.random.fold_in(ka, i))
                       for i, a in enumerate(env.agents)]
        _, state, _, _, _ = step_fn(ks, state, actions)

        img = env.render(state)
        check_frame(env_id, t, img, shape)

        if has_grid_stamps:
            grid = onp.asarray(state.grid)
            if (grid >= 1000).any():
                saw_claim = True
            for code in agent_codes:
                assert (grid == code).any(), (
                    f"{env_id} step {t}: agent code {code} missing from grid "
                    f"(agent vanished from rendering)")

    if env_id == "territory_open":
        assert saw_claim, ("territory rollout never claimed a resource; "
                           "claim-color regression not exercised — raise STEPS")

    # warm-cache timing
    t0 = time.perf_counter()
    for _ in range(WARM_TIMING_REPS):
        env.render(state)
    warm = (time.perf_counter() - t0) / WARM_TIMING_REPS
    assert warm < MAX_WARM_SECONDS, f"{env_id}: warm render {warm:.3f}s/frame"
    print(f"ok: {env_id} ({env.num_agents} agents, warm render {warm*1000:.1f} ms/frame)")


def test_cleanup_river_background():
    """Agent standing in the cleanup river must be drawn on water, not on
    the default sand background (regression: the agent grid stamp destroys
    the terrain value, so render must recover it from self.RIVER)."""
    from opensocialjax.environments.cleanup.clean_up import Items

    env = opensocialjax.make("clean_up")
    _, state = env.reset(jax.random.PRNGKey(0))

    # Move agent 0 onto the first river cell and restamp the grid.
    r, c = (int(v) for v in onp.asarray(env.RIVER)[0])
    old = onp.asarray(state.agent_locs)[0]
    code = int(onp.asarray(env._agents)[0])
    grid = state.grid.at[int(old[0]), int(old[1])].set(0).at[r, c].set(code)
    locs = state.agent_locs.at[0, 0].set(r).at[0, 1].set(c)
    state = state.replace(grid=grid, agent_locs=locs)

    img = env.render(state)
    d = int(onp.asarray(locs)[0, 2])
    # tile (r, c) after pad/crop (image matches ASCII orientation, no rot)
    region = img[(r + 1) * 32:(r + 2) * 32,
                 (c + 1) * 32:(c + 2) * 32]

    def tile(terrain):
        # blit truncates float (highlighted) tiles to uint8; mirror that
        t = env.render_tile(code, agent_dir=d, agent_hat=False,
                            highlight=True, tile_size=32, terrain=terrain)
        return t.astype(onp.uint8)

    assert onp.array_equal(region, tile(int(Items.river))), \
        "agent-in-river tile does not use the river background"
    assert not onp.array_equal(region, tile(None)), \
        "agent-in-river tile still matches the sand background"
    print("ok: cleanup river background under agent")


def test_tile_cache_isolation():
    from opensocialjax.environments.common_harvest.harvest_open import Harvest_open
    e1 = Harvest_open(num_agents=7)
    e2 = Harvest_open(num_agents=3)
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
