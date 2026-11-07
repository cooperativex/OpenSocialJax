"""RL^2 (recurrent PPO, algorithms/RPPO/rl2.py) on OpenHarvest — the game the LLM agents play.

The environment comes from the LLM harness's own constructor (llm_policy.open_harvest_policy.make_env), so a policy
trained here plays exactly the evaluated game: held-out pay / regrowth orders, the original 16 x 22 Commons Harvest
map size with six patches, patch colours drawn with replacement, individual rewards. ENV_KWARGS are make_env's
arguments; with layouts=random every meta-episode draws its map from the pool (kept across its trials), where the
harness plays one map per seed. Training plays the training splits of the rules and maps (on_train=True), the
held-out evaluation the splits the LLM agents play.
"""
import jax
import jax.numpy as jnp

from algorithms.RPPO import rl2
from opensocialjax.environments.discovery_rules import (NUM_HARVEST_HUES, NUM_RIPE_STAGES, PAY_TABLE, RATE_TABLE,
                                                        decode_harvest_ruleset)
from opensocialjax.environments.open_harvest.open_harvest import Items

# the observation's channels: item one-hot (its 15 apple codes name hue and ripeness), then 10 agent features;
# no separate colour axis and nothing held
OBS_LAYOUT = (len(Items) - 1, 0, 10, 0)


def build_env(config, on_train):
    from llm_policy.open_harvest_policy import make_env
    return make_env(**config["ENV_KWARGS"], on_train=on_train)


def policy_extras(config):
    """What each agent is shown about the step it just took, beyond its own points (rl2.make_train's `extras`):

    LAST_EATEN_INPUT   one-hots of the hue (5) and the stage (3) of the apple it ate, zeros if it ate none. The LLM
                       agents are told the same ("ATE a <hue> <stage> apple: +p points", llm_policy/open_harvest_policy.py),
                       and the points alone do not say which apple paid them: the view turns with the agent while its
                       moves are world directions, so with several apples around it cannot tell which one it stepped on.
    ORACLE_RULE_INPUT  the meta-episode's true pay table (hue x stage -> 1 / 0.5 / 0.25) and regrowth table (hue x stage
                       -> x2 / x1 / x0.5), on every step. A control, not the LLM's game: with the rule handed over, is
                       eating at the right stage learnable at all?
    Returns (dim, fn) with fn(state, info) -> (num_agents, dim), or None when both are off."""
    last = bool(config.get("LAST_EATEN_INPUT", True))
    oracle = bool(config.get("ORACLE_RULE_INPUT", False))
    if not (last or oracle):
        return None
    H, S = NUM_HARVEST_HUES, NUM_RIPE_STAGES

    def fn(state, info):
        n = state.agent_locs.shape[0]
        parts = []
        if last:
            if info is None:                      # right after a reset nothing has been eaten
                parts.append(jnp.zeros((n, H + S)))
            else:
                ate = (info["eaten"].reshape(n) > 0)[:, None]
                parts.append(ate * jnp.concatenate([jax.nn.one_hot(info["eaten_hue"].reshape(n), H),
                                                    jax.nn.one_hot(info["eaten_stage"].reshape(n), S)], axis=-1))
        if oracle:
            rate_idx, order = decode_harvest_ruleset(state.rule_encoding, H)
            rule = jnp.concatenate([PAY_TABLE[order].reshape(-1), RATE_TABLE[rate_idx].reshape(-1)])
            parts.append(jnp.broadcast_to(rule, (n, rule.shape[0])))
        return jnp.concatenate(parts, axis=-1)

    return (H + S) * last + 2 * H * S * oracle, fn


def make_train(config):
    return rl2.make_train(config, build_env(config, on_train=True), build_env(config, on_train=False), OBS_LAYOUT,
                          extras=policy_extras(config))


# Used by algorithms/train.py to dispatch through algorithms.RPPO._runner.
SINGLE_RUN_KWARGS = {"wandb_name": "rppo_cnn_open_harvest", "build_env": build_env, "obs_layout": OBS_LAYOUT,
                     "policy_extras": policy_extras}
TUNE_KWARGS = {"sweep_name": "open_harvest"}
