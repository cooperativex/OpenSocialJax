"""RL^2 (recurrent PPO, algorithms/RPPO/rl2.py) on OpenHarvest — the game the LLM agents play.

The environment comes from the LLM harness's own constructor (llm_policy.open_harvest_policy.make_env), so a policy
trained here plays exactly the evaluated game: held-out pay / regrowth orders, the original 16 x 22 Commons Harvest
map size with six patches, patch colours drawn with replacement, individual rewards. ENV_KWARGS are make_env's
arguments; with layouts=random every meta-episode draws its map from the pool (kept across its trials), where the
harness plays one map per seed. Training plays the training splits of the rules and maps (on_train=True), the
held-out evaluation the splits the LLM agents play.
"""
from algorithms.RPPO import rl2
from opensocialjax.environments.open_harvest.open_harvest import Items

# the observation's channels: item one-hot (its 15 apple codes name hue and ripeness), then 10 agent features;
# no separate colour axis and nothing held
OBS_LAYOUT = (len(Items) - 1, 0, 10, 0)


def build_env(config, on_train):
    from llm_policy.open_harvest_policy import make_env
    return make_env(**config["ENV_KWARGS"], on_train=on_train)


def make_train(config):
    return rl2.make_train(config, build_env(config, on_train=True), build_env(config, on_train=False), OBS_LAYOUT)


# Used by algorithms/train.py to dispatch through algorithms.RPPO._runner.
SINGLE_RUN_KWARGS = {"wandb_name": "rppo_cnn_open_harvest", "build_env": build_env, "obs_layout": OBS_LAYOUT}
TUNE_KWARGS = {"sweep_name": "open_harvest"}
