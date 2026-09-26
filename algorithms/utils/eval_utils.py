"""
Shared evaluation utilities for all algorithms.

This module provides evaluation functions for trained MARL policies.
Different algorithms have slightly different evaluation patterns:
- IPPO: Uses ActorCritic with PARAMETER_SHARING logic
- MAPPO/IRAT: Use Actor network (no critic in eval)
- SVO/TRANSFER: Use ActorCritic without PARAMETER_SHARING logic
All variants save a seed-tagged GIF and log it to WandB when a run is active.
"""

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from pathlib import Path
import wandb

from algorithms.utils.networks import ActorCritic, SmallActor
from algorithms.utils.data_utils import unbatchify


def evaluate_ippo(params, env, save_path, config):
    """
    Evaluation function for IPPO algorithm.

    Supports both parameter sharing and individual networks.
    Logs evaluation GIF to WandB.

    Args:
        params: Model parameters (single or list for multi-agent)
        env: Environment instance
        save_path: Path where params were saved (unused but kept for API compatibility)
        config: Configuration dictionary
    """
    rng = jax.random.PRNGKey(0)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False

    pics = []
    img = env.render(state)
    pics.append(img)

    # Extract environment name for root_dir
    env_name = config["ENV_NAME"]
    # Map environment names to evaluation directories
    env_dir_mapping = {
        "clean_up": "cleanup",
        "open_cleanup": "open_cleanup",
        "coin_game": "coins",
        "coop_mining": "coop_mining",
        "gift": "gift",
        "harvest_common_open": "harvest_common",
        "harvest_common_closed": "harvest_closed",
        "harvest_common_partnership": "harvest_partnership",
        "mushrooms": "mushrooms",
        "pd_arena": "pd_arena",
        "territory_open": "territory_open",
    }
    env_dir = env_dir_mapping.get(env_name, env_name)
    root_dir = f"evaluation/{env_dir}"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    # Build the network module once (flax modules are stateless; per-frame
    # reconstruction just wasted time) and jit the apply for the frame loop.
    network = ActorCritic(action_dim=env.action_space().n, activation=config.get("ACTIVATION", "relu"))
    apply_fn = jax.jit(network.apply)

    num_frames = int(config.get("GIF_NUM_FRAMES", 0))
    for _t in range(num_frames):
        # Use model to select actions
        if config.get("PARAMETER_SHARING", True):
            obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
            pi, _ = apply_fn(params, obs_batch)
            rng, _rng = jax.random.split(rng)
            actions = pi.sample(seed=_rng)
            # Convert action format
            env_act = {k: v.squeeze() for k, v in unbatchify(
                actions, env.agents, 1, env.num_agents
            ).items()}
        else:
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            env_act = {}
            for i in range(env.num_agents):
                obs_i = jnp.expand_dims(obs_batch[i], axis=0)
                pi, _ = apply_fn(params[i], obs_i)
                rng, _rng = jax.random.split(rng)
                single_action = pi.sample(seed=_rng)
                env_act[env.agents[i]] = single_action

        # Execute actions
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])
        done = done["__all__"]

        # Render
        img = env.render(state)
        pics.append(img)

    # Save GIF
    print(f"Saving Episode GIF")
    pics = [Image.fromarray(np.array(img)) for img in pics]
    n_agents = len(env.agents)
    gif_path = f"{root_dir}/{n_agents}-agents_seed-{config['SEED']}_frames-{len(pics) - 1}.gif"
    pics[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics[1:],
        duration=200,
        loop=0,
    )

    # Log the GIF to WandB
    if wandb.run is not None:
        print("Logging GIF to WandB")
        wandb.log({"Episode GIF": wandb.Video(gif_path, caption="Evaluation Episode", format="gif")})


def evaluate_mappo_style(params, env, save_path, config, use_actor_only=True):
    """
    Evaluation function for MAPPO/IRAT/SVO/TRANSFER algorithms.

    These algorithms use simpler evaluation without PARAMETER_SHARING logic.

    Args:
        params: Model parameters
        env: Environment instance
        save_path: Path where params were saved (unused but kept for API compatibility)
        config: Configuration dictionary
        use_actor_only: If True, use Actor network; if False, use ActorCritic (for SVO/TRANSFER)
    """
    rng = jax.random.PRNGKey(0)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False

    pics = []
    img = env.render(state)
    pics.append(img)

    # Extract environment name for root_dir
    env_name = config["ENV_NAME"]
    # Map environment names to evaluation directories
    env_dir_mapping = {
        "clean_up": "cleanup",
        "open_cleanup": "open_cleanup",
        "coin_game": "coins",
        "coop_mining": "coop_mining",
        "gift": "gift",
        "harvest_common_open": "harvest_common",
        "harvest_common_closed": "harvest_closed",
        "harvest_common_partnership": "harvest_partnership",
        "mushrooms": "mushrooms",
        "pd_arena": "pd_arena",
        "territory_open": "territory_open",
        "harvest_open": "harvest_open",
        "harvest_closed": "harvest_closed",
        "harvest_partnership": "harvest_partnership",
    }
    env_dir = env_dir_mapping.get(env_name, env_name)
    root_dir = f"evaluation/{env_dir}"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    # Build the network module once and jit the apply for the frame loop.
    if use_actor_only:
        # MAPPO uses SmallActor (features=16), not Actor (features=64)
        network = SmallActor(action_dim=env.action_space().n, activation=config.get("ACTIVATION", "relu"))
    else:
        network = ActorCritic(action_dim=env.action_space().n, activation=config.get("ACTIVATION", "relu"))
    apply_fn = jax.jit(network.apply)

    num_frames = int(config.get("GIF_NUM_FRAMES", 0))
    for _t in range(num_frames):
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)

        # Use model to select actions
        if use_actor_only:
            pi = apply_fn(params, obs_batch)
        else:
            pi, _ = apply_fn(params, obs_batch)

        rng, _rng = jax.random.split(rng)
        actions = pi.sample(seed=_rng)

        # Convert action format
        env_act = {k: v.squeeze() for k, v in unbatchify(
            actions, env.agents, 1, env.num_agents
        ).items()}

        # Execute actions
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])
        done = done["__all__"]

        # Render
        img = env.render(state)
        pics.append(img)

    # Save GIF (seed-tagged so multi-seed runs don't overwrite each other)
    print(f"Saving Episode GIF")
    pics = [Image.fromarray(np.array(img)) for img in pics]
    gif_path = f"{root_dir}/{len(env.agents)}-agents_seed-{config['SEED']}_frames-{len(pics) - 1}.gif"
    pics[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics[1:],
        duration=200,
        loop=0,
    )

    # Log the GIF to WandB
    if wandb.run is not None:
        print("Logging GIF to WandB")
        wandb.log({"Episode GIF": wandb.Video(gif_path, caption="Evaluation Episode", format="gif")})


def _obs_layout():
    from opensocialjax.environments.open_cleanup.open_cleanup import NUM_COLORS, NUM_KINDS
    return (NUM_KINDS - 1, NUM_COLORS, 10, NUM_COLORS)


def evaluate_rppo(params, env, save_path, config):
    """Evaluation for recurrent (RL^2) policies.

    Same GIF output as the feed-forward evaluators, but the hidden state has
    to be carried between frames and the policy fed its previous action and
    reward — a recurrent agent evaluated statelessly would look untrained.
    The rollout is not reset between trials, which is the point: memory built
    up in trial 1 is what should make trial 2 better.
    """
    from algorithms.utils.networks import ActorCriticRNN, ScannedRNN

    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)

    pics = [env.render(state)]

    env_name = config["ENV_NAME"]
    env_dir_mapping = {
        "clean_up": "cleanup",
        "open_cleanup": "open_cleanup",
        "coin_game": "coins",
        "coop_mining": "coop_mining",
        "gift": "gift",
        "harvest_common_open": "harvest_common",
        "mushrooms": "mushrooms",
        "pd_arena": "pd_arena",
        "territory_open": "territory_open",
    }
    root_dir = f"evaluation/{env_dir_mapping.get(env_name, env_name)}"
    Path(root_dir + "/state_pics").mkdir(parents=True, exist_ok=True)

    n_agents = len(env.agents)
    network = ActorCriticRNN(
        action_dim=env.action_space().n,
        activation=config.get("ACTIVATION", "relu"),
        hidden_size=config.get("RNN_HIDDEN_SIZE", 128),
        head_hidden_size=config.get("HEAD_HIDDEN_SIZE", 64),
        obs_layout=_obs_layout(),
        aux=float(config.get("AUX_COEF", 0.0)) > 0.0,
    )
    apply_fn = jax.jit(network.apply)

    hstate = ScannedRNN.initialize_carry(n_agents, config.get("RNN_HIDDEN_SIZE", 128))
    prev_action = jnp.zeros(n_agents, dtype=jnp.int32)
    prev_reward = jnp.zeros(n_agents)
    last_done = jnp.zeros(n_agents, dtype=bool)

    num_frames = int(config.get("GIF_NUM_FRAMES", 0))
    for _t in range(num_frames):
        obs_batch = jnp.stack([obs[a] for a in env.agents])
        hstate, pi, *_ = apply_fn(
            params,
            hstate,
            (obs_batch[None, :], prev_action[None, :],
             prev_reward[None, :], last_done[None, :]),
        )
        rng, _rng = jax.random.split(rng)
        action = pi.sample(seed=_rng).squeeze(0)

        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(
            _rng, state, [int(a) for a in action]
        )
        prev_action = action
        prev_reward = jnp.asarray(reward).reshape(-1)[:n_agents]
        last_done = jnp.full(n_agents, bool(done["__all__"]))

        pics.append(env.render(state))

    print(f"Saving Episode GIF")
    pics = [Image.fromarray(np.array(img)) for img in pics]
    gif_path = (f"{root_dir}/{n_agents}-agents_seed-{config['SEED']}"
                f"_frames-{len(pics) - 1}.gif")
    pics[0].save(
        gif_path, format="GIF", save_all=True, optimize=False,
        append_images=pics[1:], duration=200, loop=0,
    )

    if wandb.run is not None:
        print("Logging GIF to WandB")
        wandb.log({"Episode GIF": wandb.Video(
            gif_path, caption="Evaluation Episode", format="gif")})
