"""RL^2 (algorithms/RPPO) on both games: the configs build the LLM's game, a tiny run trains end to end, the
common-reward switch pays every agent its environment's total, and OpenCleanup's initial-dirt curriculum reaches
training only.

Run:
  python -m pytest tests/test_rl2.py -q
"""
import os

import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from algorithms.RPPO import rl2

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "algorithms", "RPPO", "config")
TINY = ["WANDB_MODE=disabled", "NUM_ENVS=2", "NUM_STEPS=10", "TOTAL_TIMESTEPS=40", "NUM_MINIBATCHES=2",
        "RNN_HIDDEN_SIZE=16", "HEAD_HIDDEN_SIZE=8", "EVAL_NUM_ENVS=2", "EVAL_EVERY=1",
        "ENV_KWARGS.inner=10", "ENV_KWARGS.outer=2"]


def _config(env, overrides=()):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return OmegaConf.to_container(compose(config_name=f"rppo_cnn_{env}", overrides=list(overrides)), resolve=True)


def test_common_reward_pays_every_agent_the_total():
    r = jnp.array([[1.0, 0.0, 0.5], [0.0, 0.0, 0.0]])        # 2 environments x 3 agents
    assert onp.array_equal(rl2.training_reward(r, "individual"), r)
    assert onp.array_equal(rl2.training_reward(r, "common"), [[1.5, 1.5, 1.5], [0.0, 0.0, 0.0]])
    with pytest.raises(AssertionError):
        rl2.training_reward(r, "shared")


@pytest.mark.parametrize("env_name", ["open_cleanup", "open_harvest"])
def test_the_config_is_the_llm_game(env_name):
    import importlib
    config = _config(env_name)
    assert config["REWARD"] == "individual" and config["ENV_KWARGS"]["agents"] == 5
    mod = importlib.import_module(f"algorithms.RPPO.rppo_cnn_{env_name}")
    train_env, eval_env = mod.build_env(config, on_train=True), mod.build_env(config, on_train=False)
    assert train_env.layout_pool is not None and eval_env.layout_pool is not None
    assert train_env.layout_pool.size + eval_env.layout_pool.size == 256        # disjoint 80 / 20 map splits
    assert train_env.rule_pool.size > eval_env.rule_pool.size                    # disjoint 80 / 20 rule splits
    assert (train_env.num_inner_steps, train_env.num_outer_steps) == (eval_env.num_inner_steps, eval_env.num_outer_steps)
    assert sum(mod.OBS_LAYOUT) == train_env.observation_space()[0].shape[-1]


def test_the_cleanup_curriculum_is_training_only():
    from algorithms.RPPO import rppo_cnn_open_cleanup as mod
    from opensocialjax.environments.open_cleanup.open_cleanup import is_dirt
    default = _config("open_cleanup")
    assert default["TRAIN_INITIAL_DIRT"] == 1.0
    assert mod.build_env(default, on_train=True).initial_dirt_range is None   # no curriculum unless asked for
    config = _config("open_cleanup", ["TRAIN_INITIAL_DIRT=[0.5,1.0]"])
    train_env, eval_env = mod.build_env(config, on_train=True), mod.build_env(config, on_train=False)

    def dirty_fraction(env, n=16):
        """per reset: the share of the map's waste cells that open dirty"""
        st = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), n))[1]
        dirty = onp.asarray(is_dirt(st.grid))
        out = []
        for i in range(n):
            cells = onp.unique(onp.asarray(env._LP["dirt"][int(st.layout_id[i])]), axis=0)
            out.append(dirty[i][cells[:, 0], cells[:, 1]].mean())
        return onp.array(out)

    assert (dirty_fraction(eval_env) == 1.0).all()                            # evaluation: the LLM's game
    f = dirty_fraction(train_env)
    assert f.min() >= 0.3 and f.max() <= 1.0 and f.std() > 0.05, f            # training: drawn from [0.5, 1.0]
    with pytest.raises(AssertionError):
        mod.build_env(_config("open_cleanup", ["TRAIN_INITIAL_DIRT=0.7"]), on_train=True)


@pytest.mark.parametrize("env_name,overrides", [
    ("open_cleanup", ["AUX_COEF=0.1", "TRAIN_INITIAL_DIRT=[0.5,1.0]"]),    # with the auxiliary loss and the curriculum
    ("open_harvest", ["reward=common"]),           # with common rewards
])
def test_a_tiny_run_trains(env_name, overrides):
    import importlib
    config = _config(env_name, TINY + overrides)
    out = jax.jit(importlib.import_module(f"algorithms.RPPO.rppo_cnn_{env_name}").make_train(config))(jax.random.PRNGKey(0))
    m = jax.device_get(out["metrics"])
    assert m["update_step"].tolist() == [0, 1]
    for k in ("loss/total", "loss/value", "eval/returns_mean", "eval/return_trial_1", "eval/return_trial_2"):
        assert onp.isfinite(m[k]).all(), k
    assert "eval/return_trial_3" not in m
