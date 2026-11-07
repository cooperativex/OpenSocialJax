"""RL^2 (algorithms/RPPO) on both games: the configs build the LLM's game, a tiny run trains end to end, the
common-reward switch pays every agent its environment's total (in the harness's points) while each agent is fed back
only its own points, OpenCleanup's initial-dirt curriculum reaches training only, the separate critic shares no
parameter (and no gradient) with the actor, and OpenHarvest's extra inputs say what was eaten (and, for the oracle
control, the true rule).

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


def test_rewards_are_points_and_the_policy_sees_its_own():
    # the environments pay an apple as num_agents x its value to its eater: here agent 0 ate a 1-point apple and
    # agent 2 a 0.5-point one, in an environment of 3 agents
    env_reward = jnp.array([[3.0, 0.0, 1.5], [0.0, 0.0, 0.0]])
    own = [[1.0, 0.0, 0.5], [0.0, 0.0, 0.0]]
    trained, seen = rl2.policy_rewards(env_reward, "individual")
    assert onp.allclose(trained, own) and onp.allclose(seen, own)
    trained, seen = rl2.policy_rewards(env_reward, "common")
    assert onp.allclose(trained, [[1.5, 1.5, 1.5], [0.0, 0.0, 0.0]])      # trained on the team's points
    assert onp.allclose(seen, own)                                        # but shown only its own, as the LLM is


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


def _tiny_network(separate_critic, extra_dim):
    from algorithms.utils.networks import ActorCriticRNN, ScannedRNN
    net = ActorCriticRNN(action_dim=4, hidden_size=8, head_hidden_size=8, obs_layout=(3, 0, 2, 0),
                         separate_critic=separate_critic, extra_dim=extra_dim)
    T, B = 3, 2
    x = (jnp.ones((T, B, 5, 5, 5)), jnp.zeros((T, B), jnp.int32), jnp.ones((T, B)), jnp.zeros((T, B), bool))
    if extra_dim:
        x = x + (jnp.ones((T, B, extra_dim)),)
    h0 = ScannedRNN.initialize_carry(B, ActorCriticRNN.carry_size(8, separate_critic))
    return net, net.init(jax.random.PRNGKey(0), h0, x), h0, x


def test_the_shared_network_keeps_its_parameters():
    # both options off = the network every earlier checkpoint was trained with: same parameter names
    _, params, _, _ = _tiny_network(False, 0)
    assert sorted(params["params"]) == sorted(["kind_emb", "feat_emb", "Conv_0", "Conv_1", "Conv_2", "Embed_0",
                                               "ScannedRNN_0", "actor_hidden", "actor_out", "critic_hidden",
                                               "critic_out"])


def test_the_separate_critic_shares_nothing_with_the_actor():
    net, params, h0, x = _tiny_network(True, 3)
    labels = rl2.actor_labels(params)
    crit = {k for k, v in labels["params"].items() if set(jax.tree.leaves(v)) == {"critic"}}
    assert crit == {"critic_kind_emb", "critic_feat_emb", "critic_Conv_0", "critic_Conv_1", "critic_Conv_2",
                    "critic_Embed_0", "critic_ScannedRNN_0", "critic_hidden", "critic_out"}
    h, pi, value = net.apply(params, h0, x)
    assert h.shape == (2, 16)                                                   # actor half + critic half

    def nonzero(grads, names):
        return {k for k in names if any(onp.abs(onp.asarray(g)).max() > 0 for g in jax.tree.leaves(grads["params"][k]))}
    g_value = jax.grad(lambda p: net.apply(p, h0, x)[2].sum())(params)
    g_policy = jax.grad(lambda p: net.apply(p, h0, x)[1].logits.sum())(params)
    actor_names = set(params["params"]) - crit
    assert nonzero(g_value, actor_names) == set()                               # the value loss never reaches the actor
    assert nonzero(g_policy, crit) == set()                                     # nor the policy loss the critic
    assert nonzero(g_value, crit) and nonzero(g_policy, actor_names)


def test_harvest_extras_say_what_was_eaten_and_the_oracle_the_rule():
    from algorithms.RPPO import rppo_cnn_open_harvest as HV
    from opensocialjax.environments.discovery_rules import decode_harvest_ruleset, PAY_ORDERS, STAGE_PAY, REGROW_RATES
    config = _config("open_harvest", ["ENV_KWARGS.inner=10", "ENV_KWARGS.outer=2", "ORACLE_RULE_INPUT=True"])
    dim, fn = HV.policy_extras(config)
    assert dim == 8 + 30
    env = HV.build_env(config, on_train=False)
    _, state = env.reset(jax.random.PRNGKey(3))
    n = env.num_agents
    ex0 = onp.asarray(fn(state, None))
    assert ex0.shape == (n, dim) and (ex0[:, :8] == 0).all()                    # nothing eaten before the first step
    info = {"eaten": jnp.array([1, 0, 0, 1, 0]), "eaten_hue": jnp.array([2, -1, -1, 4, -1]),
            "eaten_stage": jnp.array([1, -1, -1, 0, -1])}
    ex = onp.asarray(fn(state, info))
    assert ex[0, :8].tolist() == [0, 0, 1, 0, 0, 0, 1, 0] and ex[3, :8].tolist() == [0, 0, 0, 0, 1, 1, 0, 0]
    assert (ex[[1, 2, 4], :8] == 0).all()
    rate_idx, order = decode_harvest_ruleset(state.rule_encoding)
    pay = [[STAGE_PAY[r] for r in PAY_ORDERS[int(o)]] for o in onp.asarray(order)]
    speed = [[REGROW_RATES[r] for r in PAY_ORDERS[int(o)]] for o in onp.asarray(rate_idx)]
    assert onp.allclose(ex[:, 8:23], onp.ravel(pay)) and onp.allclose(ex[:, 23:], onp.ravel(speed))
    _, state2, _, _, real_info = env.step(jax.random.PRNGKey(4), state, [jnp.int32(6)] * n)   # a real step's info fits
    assert fn(state2, real_info).shape == (n, dim)
    assert HV.policy_extras(_config("open_harvest", ["LAST_EATEN_INPUT=False"])) is None
    assert HV.policy_extras(_config("open_harvest"))[0] == 8                    # default: what was eaten, no oracle


@pytest.mark.parametrize("env_name,overrides", [
    ("open_cleanup", ["AUX_COEF=0.1", "TRAIN_INITIAL_DIRT=[0.5,1.0]"]),    # with the auxiliary loss and the curriculum
    ("open_harvest", ["reward=common"]),           # with common rewards (and the default what-was-eaten input)
    ("open_harvest", ["reward=common", "SEPARATE_CRITIC=True", "ORACLE_RULE_INPUT=True"]),   # separate critic, oracle
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
