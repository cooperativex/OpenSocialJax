"""Shared dataclasses/NamedTuples used across algorithms."""

import jax.numpy as jnp
from typing import NamedTuple


class Transition(NamedTuple):
    """Standard PPO transition (IPPO / SVO / TRANSFER)."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class MAPPOTransition(NamedTuple):
    """Centralized-critic transition for MAPPO (adds global_done & world_state)."""
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray


class IRATTransition(NamedTuple):
    """IRAT dual-policy transition with separate individual/team heads."""
    global_done: jnp.ndarray
    done: jnp.ndarray
    # Individual policy
    ind_action: jnp.ndarray
    ind_value: jnp.ndarray
    ind_log_prob: jnp.ndarray
    # Team policy
    team_action: jnp.ndarray
    team_value: jnp.ndarray
    team_log_prob: jnp.ndarray
    # Rewards
    ind_reward: jnp.ndarray
    team_reward: jnp.ndarray
    # Observations
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray


class RecurrentTransition(NamedTuple):
    """PPO transition for recurrent (RL^2-style) policies.

    Two done flags are kept on purpose. `done` is the flag produced *by* this
    step and is what GAE bootstraps against; `last_done` is the flag that was
    already true when the step began and is the RNN's reset signal. Storing
    only one and reusing it for both is an off-by-one that silently lets
    memory leak across episode boundaries during the update.

    `prev_action` / `prev_reward` are the RL^2 inputs — what the agent did on
    the previous step and what it got for it.
    """
    done: jnp.ndarray
    last_done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    prev_action: jnp.ndarray
    prev_reward: jnp.ndarray
    info: jnp.ndarray
