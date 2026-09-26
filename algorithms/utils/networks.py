"""
Shared neural network architectures for multi-agent reinforcement learning algorithms.

This module contains CNN-based network architectures used across different MARL algorithms
including IPPO, MAPPO, SVO, Inequity Aversion, and AAA.
"""

import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal, glorot_normal, zeros_init
import math
from typing import Sequence
import distrax
import jax.numpy as jnp


class CNN(nn.Module):
    """
    Convolutional Neural Network for visual feature extraction.

    Architecture:
        - Conv2D (32 filters, 5x5 kernel)
        - Conv2D (32 filters, 3x3 kernel)
        - Conv2D (32 filters, 3x3 kernel)
        - Dense (64 units)

    All layers use orthogonal initialization with sqrt(2) scaling.

    Attributes:
        activation: Activation function name ("relu" or "tanh")
    """
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for IPPO, SVO, and Inequity Aversion algorithms.

    This network uses a shared CNN backbone with separate actor and critic heads.
    Used in algorithms that don't require separate actor/critic networks.

    Architecture:
        - Shared CNN feature extractor
        - Actor head: Dense(64) -> Dense(action_dim) -> Categorical distribution
        - Critic head: Dense(64) -> Dense(1) -> Value estimate

    Attributes:
        action_dim: Number of discrete actions
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Tuple of (policy distribution, value estimate)
    """
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)

        # Actor head
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic head
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Actor(nn.Module):
    """
    Standalone Actor network for MAPPO algorithm.

    MAPPO uses separate actor and critic networks to allow independent parameter updates.
    The actor takes per-agent observations and outputs action distributions.

    Architecture:
        - CNN feature extractor
        - Dense(64) -> Dense(action_dim) -> Categorical distribution

    Attributes:
        action_dim: Number of discrete actions
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Categorical policy distribution over actions
    """
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(obs)

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        return pi


class Critic(nn.Module):
    """
    Standalone Critic network for MAPPO algorithm.

    MAPPO critic takes world state (concatenated observations from all agents)
    as input to estimate centralized value function.

    Architecture:
        - CNN feature extractor (processes world state)
        - Dense(64) -> Dense(1) -> Value estimate

    Attributes:
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Scalar value estimate (squeezed to remove last dimension)
    """
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        world_state = x

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(world_state)

        hidden = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        hidden = activation(hidden)

        value = nn.Dense(
            features=1,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(hidden)

        # Squeeze to remove last dimension
        return jnp.squeeze(value, axis=-1)


# ============================================================================
# MAPPO Small Network Architectures (features=16)
# ============================================================================
# MAPPO uses smaller networks compared to IPPO/SVO for efficiency


class SmallCNN(nn.Module):
    """
    Small Convolutional Neural Network for MAPPO algorithm.

    This is a lighter version of CNN used specifically by MAPPO for faster training
    with reduced model capacity.

    Architecture:
        - Conv2D (16 filters, 3x3 kernel) - single conv layer
        - Dense (16 units)

    All layers use orthogonal initialization with sqrt(2) scaling.

    Attributes:
        activation: Activation function name ("relu" or "tanh")
    """
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=16,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class SmallActor(nn.Module):
    """
    Small Actor network for MAPPO algorithm.

    Uses SmallCNN backbone with reduced hidden layer size (16 instead of 64).
    Designed for faster training in multi-agent scenarios.

    Architecture:
        - SmallCNN feature extractor
        - Dense(16) -> Dense(action_dim) -> Categorical distribution

    Attributes:
        action_dim: Number of discrete actions
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Categorical policy distribution over actions
    """
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = SmallCNN(self.activation)(obs)

        actor_mean = nn.Dense(
            16, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        return pi


class SmallCritic(nn.Module):
    """
    Small Critic network for MAPPO algorithm.

    Uses SmallCNN backbone with reduced hidden layer size (16 instead of 64).
    Processes world state (concatenated agent observations) for centralized value estimation.

    Architecture:
        - SmallCNN feature extractor (processes world state)
        - Dense(16) -> Dense(1) -> Value estimate

    Attributes:
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Scalar value estimate (squeezed to remove last dimension)
    """
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        world_state = x

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = SmallCNN(self.activation)(world_state)

        hidden = nn.Dense(
            features=16,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        hidden = activation(hidden)

        value = nn.Dense(
            features=1,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(hidden)

        # Squeeze to remove last dimension
        return jnp.squeeze(value, axis=-1)




class ScannedRNN(nn.Module):
    """GRU core scanned over a leading time axis, with per-step resets.

    The reset signal is the `done` flag that was true *before* each step, so a
    sequence that crosses an episode boundary starts from a fresh carry
    instead of leaking memory across episodes. This is what lets a whole
    rollout be replayed in one call during the PPO update.

    Inputs are `(carry, (x, resets))` with `x` shaped `(T, B, F)` and `resets`
    `(T, B)`; the carry is `(B, hidden_size)`.
    """
    hidden_size: int = 128

    @nn.compact
    def __call__(self, carry, xs):
        # nn.scan so the GRU parameters are shared across the time axis
        scan = nn.scan(
            lambda module, carry, x: module(carry, x),
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )
        return scan(_GRUStep(self.hidden_size), carry, xs)

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size))


class _GRUStep(nn.Module):
    """One GRU step with a reset gate on the carry (used by ScannedRNN).

    The cell is xland-minigrid's ``training/nn.py::GRU`` (PyTorch-style
    gates with the ``bn`` bias inside the reset-gated term, ``Wi`` glorot,
    ``Wh`` orthogonal), stepped one timestep at a time so the carry can be
    zeroed on episode boundaries — the one thing their batched scan does not
    need, because they reset hidden states outside the scan.
    """
    hidden_size: int

    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        carry = jnp.where(resets[:, None], jnp.zeros_like(carry), carry)
        h = self.hidden_size
        Wi = self.param("Wi", glorot_normal(in_axis=1, out_axis=0), (h * 3, ins.shape[-1]))
        Wh = self.param("Wh", orthogonal(column_axis=0), (h * 3, h))
        bi = self.param("bi", zeros_init(), (h * 3,))
        bn = self.param("bn", zeros_init(), (h,))
        igates = jnp.split(ins @ Wi.T + bi, 3, axis=-1)
        hgates = jnp.split(carry @ Wh.T, 3, axis=-1)
        reset = nn.sigmoid(igates[0] + hgates[0])
        update = nn.sigmoid(igates[1] + hgates[1])
        new = nn.tanh(igates[2] + reset * (hgates[2] + bn))
        next_h = (1 - update) * new + update * carry
        return next_h, next_h


class ActorCriticRNN(nn.Module):
    """Recurrent actor-critic for RL^2-style meta-learning — xland-minigrid's
    ``training/nn.py::ActorCriticRNN`` on the factorised observation.

    Per cell the observation carries a kind one-hot and a colour one-hot (the
    environment's two XLand axes) plus a few agent features; XLand embeds
    its ``(tile, color)`` ids with ``nn.Embed``, and an embedding of an id is
    exactly a bias-free Dense on its one-hot, so the per-axis ``Dense(16)``
    below *is* their ``EmbeddingEncoder``. Then their conv stack (three 2x2
    VALID convs, 16/32/64, relu), flatten, and a concat with the global
    inputs: the held tool (our analogue of their direction vector,
    ``Dense(16)``), the previous action (``Embed(16)``) and the previous
    reward. The core is their GRU; the heads are theirs too
    (``Dense(head) -> tanh -> Dense(out)``, orthogonal(2)/(0.01)/(1.0)).

    Call signature: ``(hstate, (obs, prev_action, prev_reward, resets))``
    with a leading time axis; returns ``(hstate, pi, value)`` or, with
    ``aux=True``, ``(hstate, pi, value, aux_logit)``.

    ``obs_layout`` = (kind channels, colour channels, agent-feature
    channels, held-tool channels), in the order the environment lays them
    out along the last observation axis.
    """
    action_dim: Sequence[int]
    activation: str = "relu"          # kept for config compatibility; the
                                      # XLand heads use tanh regardless
    hidden_size: int = 1024
    action_embed_dim: int = 16
    obs_emb_dim: int = 16
    head_hidden_size: int = 256
    # (kind, colour, agent-feature, held-tool) channel counts — the default
    # is open_cleanup's layout: 10 non-empty kinds, NUM_COLORS colours,
    # 10 agent features, NUM_COLORS held-tool channels
    obs_layout: Sequence[int] = (10, 10, 10, 10)
    aux: bool = False

    @nn.compact
    def __call__(self, hstate, x):
        obs, prev_action, prev_reward, resets = x
        t, b = obs.shape[0], obs.shape[1]
        n_kind, n_col, n_feat, n_held = self.obs_layout
        assert obs.shape[-1] == n_kind + n_col + n_feat + n_held, (
            f"obs has {obs.shape[-1]} channels, layout expects "
            f"{n_kind}+{n_col}+{n_feat}+{n_held}")

        cells = obs.reshape((t * b,) + obs.shape[2:]).astype(jnp.float32)
        kind = cells[..., :n_kind]
        colour = cells[..., n_kind:n_kind + n_col]
        feats = cells[..., n_kind + n_col:n_kind + n_col + n_feat]
        # the held tool is broadcast into every cell; read it once
        held = cells[:, 0, 0, n_kind + n_col + n_feat:]

        # EmbeddingEncoder: one embedding per axis, concatenated per cell
        emb = jnp.concatenate([
            nn.Dense(self.obs_emb_dim, name="kind_emb")(kind),
            nn.Dense(self.obs_emb_dim, name="colour_emb")(colour),
            nn.Dense(self.obs_emb_dim, name="feat_emb")(feats),
        ], axis=-1)
        h = emb
        for width in (16, 32, 64):
            h = nn.Conv(width, (2, 2), padding="VALID",
                        kernel_init=orthogonal(math.sqrt(2)))(h)
            h = nn.relu(h)
        obs_emb = h.reshape(t, b, -1)

        held_emb = nn.Dense(self.action_embed_dim, name="held_emb")(
            held).reshape(t, b, -1)
        act_emb = nn.Embed(
            num_embeddings=self.action_dim, features=self.action_embed_dim,
        )(prev_action.astype(jnp.int32))

        rnn_in = jnp.concatenate(
            [obs_emb, held_emb, act_emb, prev_reward[..., None]], axis=-1)
        hstate, rnn_out = ScannedRNN(self.hidden_size)(hstate, (rnn_in, resets))

        def head(out_dim, final_scale, name):
            z = nn.Dense(self.head_hidden_size, kernel_init=orthogonal(2),
                         name=f"{name}_hidden")(rnn_out)
            z = nn.tanh(z)
            return nn.Dense(out_dim, kernel_init=orthogonal(final_scale),
                            name=f"{name}_out")(z)

        logits = head(self.action_dim, 0.01, "actor")
        pi = distrax.Categorical(logits=logits)
        value = jnp.squeeze(head(1, 1.0, "critic"), axis=-1)

        if self.aux:
            aux_logit = jnp.squeeze(head(1, 1.0, "aux"), axis=-1)
            return hstate, pi, value, aux_logit
        return hstate, pi, value
