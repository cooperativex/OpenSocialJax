"""Shared RPPO (recurrent RL^2 PPO) runner: single_run / tune glue, factored out of the 9 per-env files.

Each algorithms/RPPO/rppo_cnn_<env>.py defines its own `make_train(config)` (the
training loop, unchanged from the original per-env code) and then delegates to
single_run() or tune() here, passing env-specific strings as kwargs:

    @hydra.main(version_base=None, config_path="config", config_name="rppo_cnn_open_cleanup")
    def main(config):
        if config["TUNE"]:
            tune(config, make_train, sweep_name="open_cleanup")
        else:
            single_run(config, make_train, wandb_name="rppo_cnn_open_cleanup")
"""
import copy
from pathlib import Path

import jax
from omegaconf import OmegaConf
import wandb

import opensocialjax
from algorithms.utils import save_params, load_params, evaluate_rppo as evaluate



def _save_metrics(metrics, name):
    """Store per-update learning curves as an npz next to the checkpoints.

    Leaves arrive as (num_seeds, num_updates, num_actors); everything but the
    update axis is averaged, so the file stays small and plottable without a
    wandb account.
    """
    if metrics is None:
        return
    import numpy as onp

    out = {}
    for key, value in metrics.items():
        arr = onp.asarray(jax.device_get(value))
        if arr.ndim >= 2:
            arr = arr.reshape(arr.shape[0], arr.shape[1], -1).mean(axis=-1)
        out[key] = arr
    path = Path("evaluation/curves")
    path.mkdir(parents=True, exist_ok=True)
    onp.savez_compressed(path / f"{name}.npz", **out)
    print(f"** Saved learning curves -> evaluation/curves/{name}.npz **")


def single_run(config, make_train, *, wandb_name):
    """One training run, saving + evaluating at the end."""
    config = OmegaConf.to_container(config)

    # Reward suffix lets common/individual runs of the same env coexist on disk
    # and in wandb. Hidden behind .get() so the runner still works for any
    # legacy yaml that doesn't define REWARD.
    reward = config.get("REWARD")
    suffix = f"_reward_{reward}" if reward else ""

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["RPPO", "RNN", "RL2"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"{wandb_name}{suffix}",
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}{suffix}'

    # Keep the learning curves. Without this the metrics the training loop
    # already produced are thrown away, and a run that collapses part-way
    # leaves no way to see when or how — which is exactly what happened.
    _save_metrics(out.get("metrics"), f"rppo_{filename}")
    train_state = jax.tree.map(lambda x: x[0], out["runner_state"][0])
    save_path = f"./checkpoints/individual/{filename}.pkl"
    # RPPO is shared-policy only, so there is always exactly one checkpoint.
    # NB: the 'indvidual' typo is the existing on-disk convention.
    save_path = f"./checkpoints/indvidual/{filename}.pkl"
    save_params(train_state, save_path)
    params = load_params(save_path)
    evaluate(params, opensocialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), save_path, config)


def tune(default_config, make_train, *, sweep_name):
    """Hyperparameter sweep with wandb."""
    default_config = OmegaConf.to_container(default_config)

    sweep_config = {
        "name": sweep_name,
        "method": "grid",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            # "LR": {"values": [0.001, 0.0005, 0.0001, 0.00005]},
            # "ACTIVATION": {"values": ["relu", "tanh"]},
            # "UPDATE_EPOCHS": {"values": [2, 4, 8]},
            # "NUM_MINIBATCHES": {"values": [4, 8, 16, 32]},
            # "CLIP_EPS": {"values": [0.1, 0.2, 0.3]},
            # "ENT_COEF": {"values": [0.001, 0.01, 0.1]},
            # "NUM_STEPS": {"values": [64, 128, 256]},
            # "ENV_KWARGS.svo_w": {"values": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
            # "ENV_KWARGS.svo_ideal_angle_degrees": {"values": [0, 45, 90]},
            "SEED": {"values": [42, 52, 62]},
        },
    }

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])
        config = copy.deepcopy(default_config)
        # only overwrite the single nested key we're sweeping
        for k, v in dict(wandb.config).items():
            if "." in k:
                parent, child = k.split(".", 1)
                config[parent][child] = v
            else:
                config[k] = v

        run_name = f"sweep_{config['ENV_NAME']}_seed{config['SEED']}"
        wandb.run.name = run_name
        print("Running experiment:", run_name)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))
        train_state = jax.tree.map(lambda x: x[0], outs["runner_state"][0])

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)
