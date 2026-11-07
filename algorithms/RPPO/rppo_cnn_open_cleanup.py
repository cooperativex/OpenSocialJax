"""RL^2 (recurrent PPO, algorithms/RPPO/rl2.py) on OpenCleanup — the game the LLM agents play.

The environment comes from the LLM harness's own constructor (llm_policy.open_cleanup_policy.make_env), so a policy
trained here plays exactly the evaluated game: held-out attribute-craft recipes, explicit pick-up, the mini map or
the random map pool, individual rewards. ENV_KWARGS are make_env's arguments. Training plays the training splits of
the recipes and maps (on_train=True), the held-out evaluation the splits the LLM agents play. The only differences
from the harness: report_beam=False (it changes nothing in the game, only drops per-beam diagnostics from `info`),
and TRAIN_INITIAL_DIRT, the optional training curriculum (evaluation always opens the river fully polluted).
"""
from algorithms.RPPO import rl2
from opensocialjax.environments.open_cleanup.open_cleanup import NUM_COLORS, NUM_KINDS

# the observation's channels: kind one-hot, colour one-hot, 10 agent features, held-tool colour one-hot
OBS_LAYOUT = (NUM_KINDS - 1, NUM_COLORS, 10, NUM_COLORS)


def initial_dirt(config):
    """TRAIN_INITIAL_DIRT as make_env takes it: 1.0 (the evaluated game) or a (lo, hi) range drawn per trial."""
    d = config.get("TRAIN_INITIAL_DIRT", 1.0)
    if isinstance(d, (list, tuple)):
        lo, hi = (float(v) for v in d)
        assert 0.0 <= lo <= hi <= 1.0, d
        return (lo, hi)
    assert float(d) == 1.0, f"TRAIN_INITIAL_DIRT is 1.0 or a [lo, hi] range, not {d}"
    return 1.0


def build_env(config, on_train):
    from llm_policy.open_cleanup_policy import make_env
    return make_env(**config["ENV_KWARGS"], on_train=on_train, report_beam=False,
                    initial_dirt=initial_dirt(config) if on_train else 1.0)


def aux_target(info):
    """AUX_COEF target: does this step's cleaning shot clear its target? Graded only on steps where the agent fired."""
    return (info["cleaned_by_agent"] > 0).astype("float32"), info["clean_action_info"].astype("float32")


def make_train(config):
    return rl2.make_train(config, build_env(config, on_train=True), build_env(config, on_train=False),
                          OBS_LAYOUT, aux_target)


# Used by algorithms/train.py to dispatch through algorithms.RPPO._runner.
SINGLE_RUN_KWARGS = {"wandb_name": "rppo_cnn_open_cleanup", "build_env": build_env, "obs_layout": OBS_LAYOUT}
TUNE_KWARGS = {"sweep_name": "open_cleanup"}
