"""Checks for the recurrent (RL^2) PPO plumbing.

Two things in RPPO fail silently rather than loudly, so they get tests:

1. Minibatching must split the ACTOR axis and leave time intact. Reusing the
   feed-forward loop's flatten-and-shuffle would still run, still produce
   gradients, and quietly train the GRU on shredded fragments.
2. The GRU must carry memory across steps and drop it exactly on resets —
   otherwise a meta-episode's trials do not share evidence, which is the only
   reason the recurrent policy exists.

Run:
  export PYTHONPATH=$PWD:$PYTHONPATH
  JAX_PLATFORMS=cpu python tests/test_rppo_sequences.py
"""

import sys

import jax
import jax.numpy as jnp
import numpy as onp

from algorithms.RPPO.rppo_cnn_open_cleanup import split_sequence_minibatches
from algorithms.utils.networks import ActorCriticRNN, ScannedRNN


def test_minibatches_keep_each_actor_sequence_whole():
    """Every minibatch row must be one actor's contiguous trajectory, paired
    with that same actor's starting hidden state."""
    T, actors, n_mb, hidden = 6, 12, 3, 4
    per_mb = actors // n_mb

    # value[t, a] = a * 100 + t, so an actor id and a timestep are readable
    # straight off the tensor
    data = (jnp.arange(actors)[None, :] * 100 + jnp.arange(T)[:, None]).astype(jnp.int32)
    # hstate[a] = a, so a mismatched pairing is visible too
    hstate = jnp.tile(jnp.arange(actors)[:, None], (1, hidden)).astype(jnp.float32)

    perm = jax.random.permutation(jax.random.PRNGKey(0), actors)
    hs_mb, (data_mb,) = split_sequence_minibatches(
        perm, hstate, (data,), n_mb, per_mb
    )

    assert data_mb.shape == (n_mb, T, per_mb), data_mb.shape
    assert hs_mb.shape == (n_mb, per_mb, hidden), hs_mb.shape

    perm_np = onp.asarray(perm)
    for m in range(n_mb):
        for j in range(per_mb):
            actor = int(perm_np[m * per_mb + j])
            column = onp.asarray(data_mb[m, :, j])
            expected = actor * 100 + onp.arange(T)
            assert onp.array_equal(column, expected), (
                f"minibatch {m} row {j}: expected actor {actor}'s timeline "
                f"{expected.tolist()}, got {column.tolist()} — the time axis "
                f"was shuffled or actors were interleaved")
            assert onp.allclose(onp.asarray(hs_mb[m, j]), actor), (
                f"minibatch {m} row {j}: hidden state belongs to actor "
                f"{onp.asarray(hs_mb[m, j])[0]:.0f}, trajectory to {actor}")

    # every actor appears exactly once across the minibatches
    seen = sorted(int(onp.asarray(data_mb[m, 0, j]) // 100)
                  for m in range(n_mb) for j in range(per_mb))
    assert seen == list(range(actors)), f"actors lost or duplicated: {seen}"
    print(f"ok: {n_mb} minibatches x {per_mb} actors keep whole {T}-step sequences")


def test_replaying_a_split_rollout_matches_the_unsplit_one():
    """Replaying the minibatches through the GRU must reproduce exactly what a
    single pass over the full rollout produces — this is what makes the PPO
    update consistent with the data collection."""
    T, actors, n_mb, hidden, obs_dim = 5, 8, 2, 16, (5, 5, 3)
    per_mb = actors // n_mb

    key = jax.random.PRNGKey(1)
    k_obs, k_init, k_perm = jax.random.split(key, 3)
    obs = jax.random.normal(k_obs, (T, actors) + obs_dim)
    prev_a = jnp.zeros((T, actors), dtype=jnp.int32)
    prev_r = jnp.zeros((T, actors))
    resets = jnp.zeros((T, actors), dtype=bool)

    net = ActorCriticRNN(action_dim=4, hidden_size=hidden)
    hstate = ScannedRNN.initialize_carry(actors, hidden)
    params = net.init(k_init, hstate, (obs, prev_a, prev_r, resets))
    _, _, value_full = net.apply(params, hstate, (obs, prev_a, prev_r, resets))

    perm = jax.random.permutation(k_perm, actors)
    hs_mb, (obs_mb, pa_mb, pr_mb, rs_mb) = split_sequence_minibatches(
        perm, hstate, (obs, prev_a, prev_r, resets), n_mb, per_mb
    )

    perm_np = onp.asarray(perm)
    for m in range(n_mb):
        _, _, value_mb = net.apply(
            params, hs_mb[m], (obs_mb[m], pa_mb[m], pr_mb[m], rs_mb[m])
        )
        for j in range(per_mb):
            actor = int(perm_np[m * per_mb + j])
            assert onp.allclose(
                onp.asarray(value_mb[:, j]), onp.asarray(value_full[:, actor]),
                atol=1e-5), (
                f"minibatch {m} row {j} (actor {actor}) replayed to different "
                f"values than the full rollout — the split broke the sequence")
    print("ok: replaying the split rollout reproduces the full-rollout values")


def test_memory_persists_between_resets():
    """Memory must survive a step without a reset and vanish with one."""
    T, B, hidden = 4, 2, 8
    net = ActorCriticRNN(action_dim=4, hidden_size=hidden)
    h0 = ScannedRNN.initialize_carry(B, hidden)
    obs = jnp.ones((T, B, 5, 5, 3))
    pa = jnp.zeros((T, B), dtype=jnp.int32)
    pr = jnp.zeros((T, B))
    params = net.init(jax.random.PRNGKey(0), h0,
                      (obs, pa, pr, jnp.zeros((T, B), dtype=bool)))

    no_reset = jnp.zeros((T, B), dtype=bool)
    h_carried, _, _ = net.apply(params, h0, (obs, pa, pr, no_reset))
    assert not onp.allclose(onp.asarray(h_carried), 0.0), \
        "hidden state stayed at its initial value — the GRU is not accumulating"

    # a reset on the final step must wipe everything before it
    reset_last = no_reset.at[T - 1].set(True)
    h_reset, _, _ = net.apply(params, h0, (obs, pa, pr, reset_last))
    h_one_step, _, _ = net.apply(params, h0, (obs[-1:], pa[-1:], pr[-1:],
                                              jnp.zeros((1, B), dtype=bool)))
    assert onp.allclose(onp.asarray(h_reset), onp.asarray(h_one_step), atol=1e-6), \
        "a reset did not clear memory from before the episode boundary"
    print("ok: memory persists across steps and is cleared exactly on resets")


if __name__ == "__main__":
    failures = []
    for fn in (test_minibatches_keep_each_actor_sequence_whole,
               test_replaying_a_split_rollout_matches_the_unsplit_one,
               test_memory_persists_between_resets):
        try:
            fn()
        except AssertionError as e:
            failures.append(f"{fn.__name__}: {e}")
            print(f"FAIL: {fn.__name__}: {e}")
    if failures:
        sys.exit(f"{len(failures)} RPPO check(s) failed")
    print("ALL RPPO SEQUENCE CHECKS PASSED")
