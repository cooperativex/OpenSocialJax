"""Recurrent IPPO (RL^2) — PPO with a GRU policy for hidden-rule environments.

Feed-forward PPO cannot solve a task whose rule is hidden and resampled per
meta-episode: with no memory it can only learn a fixed prior over rules. This
is the RL^2 setup (Duan et al., 2016; Wang et al., 2016) that XLand-MiniGrid
uses for its meta-tasks — a recurrent actor-critic that also sees the previous
action and previous reward, so "I tried that and it did not pay" is available
as evidence within an episode.

Two things differ from the feed-forward IPPO loop, and both matter:

1. The hidden state is carried through the rollout and reset on episode
   boundaries. In open_cleanup `done` only fires at the end of a
   meta-episode, so memory persists across the trials that share a rule —
   which is the whole point.
2. Minibatching splits the **actor** axis only and keeps the time axis intact,
   because the update replays each sequence through the GRU. The feed-forward
   loop flattens (time x actor) and shuffles, which would destroy the
   sequences.

Parameter sharing only: one recurrent policy drives every agent, each with its
own hidden state. That is the usual RL^2 multi-agent setup and keeps the
sequence bookkeeping tractable.
"""

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import opensocialjax
from opensocialjax.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
import wandb

from algorithms.utils import (
    ActorCriticRNN,
    ScannedRNN,
    batchify,
    batchify_dict,
    unbatchify,
    save_params,
    load_params,
    RecurrentTransition,
)
from algorithms.utils.eval_utils import evaluate_rppo as evaluate
from opensocialjax.environments.rule_benchmark import (
    build_benchmark,
    enumerate_benchmark,
    split_from_config,
)
from opensocialjax.environments.open_cleanup.open_cleanup import (
    NUM_COLORS,
    NUM_DIRT_TYPES,
    NUM_KINDS,
    all_rulesets,
)


def split_sequence_minibatches(permutation, init_hstate, batch,
                               num_minibatches, actors_per_minibatch):
    """Split a rollout into minibatches of whole sequences.

    `batch` leaves are `(T, actors, ...)` and `init_hstate` is
    `(actors, hidden)`. Actors are permuted and dealt into minibatches; the
    time axis is never touched, so each row of a minibatch is one actor's
    contiguous trajectory and lines up with that actor's starting hidden
    state. Getting this wrong does not raise — it silently trains the GRU on
    shuffled fragments — hence tests/test_rppo_sequences.py.

    Returns `(hstate_minibatches, batch_minibatches)` with leading axis
    `num_minibatches`, batch leaves shaped `(n_mb, T, actors_per_mb, ...)`.
    """
    hstate_mb = jnp.take(init_hstate, permutation, axis=0).reshape(
        (num_minibatches, actors_per_minibatch) + init_hstate.shape[1:]
    )
    shuffled = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=1), batch
    )
    batch_mb = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(
            x.reshape((x.shape[0], num_minibatches, actors_per_minibatch)
                      + x.shape[2:]),
            0, 1,
        ),
        shuffled,
    )
    return hstate_mb, batch_mb


def make_train(config):
    # XLand-style protocol: pre-generate the rule benchmark, split it once,
    # train on one half and evaluate on the other. Without the split a policy
    # that has seen every rule thousands of times scores well with no
    # inference at all, and "did it adapt?" cannot be answered.
    # RULE_SPLIT_MODE=predicate cuts by meaning instead of by index — the
    # held-out (waste, tool) pairing never occurs in training at all.
    # chain_depth>=2 switches to XLand's offline recipe: a sampled, deduped
    # benchmark under a fixed seed (the space is no longer enumerable)
    _ek = config["ENV_KWARGS"]
    bench = build_benchmark(
        chain_depth=int(_ek.get("chain_depth", 1)),
        p_chain=float(_ek.get("p_chain", 0.5)),
        bench_seed=int(config.get("BENCH_SEED", 0)),
        bench_size=int(config.get("BENCH_SIZE", 100_000)),
        bijective=bool(_ek.get("bijective_rules", False)),
        craft=bool(_ek.get("craft_rules", False)),
        craft_tools=int(_ek.get("craft_tools", 4)),
        craft_len=int(_ek.get("craft_len", 3)),
    )
    train_rules, test_rules = split_from_config(bench, config, NUM_DIRT_TYPES)

    env = opensocialjax.make(config["ENV_NAME"],
                             rule_pool=train_rules, **config["ENV_KWARGS"])
    # Same environment, held-out rules only. Unwrapped: evaluation reads
    # returns directly rather than through the training logger.
    #
    # The training curriculum is explicitly NOT inherited: initial_dirt is
    # pinned to 1.0 so evaluation always happens in the regime the design gate
    # was measured in. Letting the easy trials leak into the eval set would
    # inflate the score with returns that never required the rule.
    eval_kwargs = dict(config["ENV_KWARGS"])
    eval_kwargs["initial_dirt"] = 1.0
    eval_env = opensocialjax.make(config["ENV_NAME"],
                                  rule_pool=test_rules, **eval_kwargs)
    if not config.get("PARAMETER_SHARING", True):
        raise ValueError(
            "RPPO implements the shared-policy RL^2 setup; set "
            "PARAMETER_SHARING=True (per-agent recurrent policies would need "
            "separate hidden-state bookkeeping)."
        )
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    # a minibatch is a set of whole sequences, so it is measured in actors
    config["NUM_ACTORS_PER_MINIBATCH"] = (
        config["NUM_ACTORS"] // config["NUM_MINIBATCHES"]
    )
    assert config["NUM_ACTORS"] % config["NUM_MINIBATCHES"] == 0, (
        f"NUM_ACTORS ({config['NUM_ACTORS']}) must divide by NUM_MINIBATCHES "
        f"({config['NUM_MINIBATCHES']}) — minibatches split actors, not steps"
    )

    env = LogWrapper(env, replace_info=False)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
                      ) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        use_aux = float(config.get("AUX_COEF", 0.0)) > 0.0
        network = ActorCriticRNN(
            env.action_space().n,
            activation=config["ACTIVATION"],
            hidden_size=config["RNN_HIDDEN_SIZE"],
            head_hidden_size=config.get("HEAD_HIDDEN_SIZE", 64),
            # the environment's channel layout: kind one-hot, colour one-hot,
            # 10 agent features, held-tool colour one-hot
            obs_layout=(NUM_KINDS - 1, NUM_COLORS, 10, NUM_COLORS),
            aux=use_aux,
        )
        rng, _rng = jax.random.split(rng)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["RNN_HIDDEN_SIZE"]
        )
        init_x = (
            jnp.zeros((1, config["NUM_ACTORS"], *env.observation_space()[0].shape)),
            jnp.zeros((1, config["NUM_ACTORS"]), dtype=jnp.int32),
            jnp.zeros((1, config["NUM_ACTORS"])),
            jnp.zeros((1, config["NUM_ACTORS"]), dtype=bool),
        )
        network_params = network.init(_rng, init_hstate, init_x)
        # Warm start (curriculum, XLand's trivial -> small staging): replace the
        # fresh parameters with a checkpoint trained on an easier variant of
        # the same environment. Shapes must match exactly — the checkpoint
        # is loaded host-side once, before the training function is jitted.
        warm = config.get("WARM_START", "")
        if warm:
            loaded = load_params(warm)
            loaded = loaded["params"] if isinstance(loaded, dict) and "params" in loaded and "params" not in network_params else loaded
            fresh_shapes = jax.tree.map(lambda x: x.shape, network_params)
            load_shapes = jax.tree.map(lambda x: x.shape, loaded)
            assert fresh_shapes == load_shapes, (
                f"WARM_START checkpoint {warm} does not match this network")
            network_params = jax.tree.map(jnp.asarray, loaded)
            print(f"[warm start] initialised from {warm}")

        adam_eps = config.get("ADAM_EPS", 1e-5)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=adam_eps),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=adam_eps),
            )
        train_state = TrainState.create(apply_fn=network.apply,
                                        params=network_params, tx=tx)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # RL^2 inputs start empty: no previous action, no previous reward
        init_prev_action = jnp.zeros(config["NUM_ACTORS"], dtype=jnp.int32)
        init_prev_reward = jnp.zeros(config["NUM_ACTORS"])
        init_done = jnp.zeros(config["NUM_ACTORS"], dtype=bool)

        def _eval_on_held_out(params, key):
            """Roll out whole meta-episodes on rules never trained on.

            Returns per-episode return, so the caller can report the spread
            as well as the mean — XLand logs the 20th percentile because a
            meta-policy that solves the easy rules and fails the rest has a
            respectable mean and a terrible tail."""
            n_agents = eval_env.num_agents
            ep_steps = (config["ENV_KWARGS"]["num_inner_steps"]
                        * config["ENV_KWARGS"]["num_outer_steps"])

            def one_episode(key):
                key, kr = jax.random.split(key)
                obs, state = eval_env.reset(kr)

                def step_fn(carry, _):
                    state, obs, hstate, prev_a, prev_r, last_d, key = carry
                    obs_b = jnp.stack([obs[a] for a in eval_env.agents])
                    hstate, pi, *_ = network.apply(
                        params, hstate,
                        (obs_b[None, :], prev_a[None, :],
                         prev_r[None, :], last_d[None, :]))
                    key, ka, ks = jax.random.split(key, 3)
                    action = pi.sample(seed=ka).squeeze(0)
                    obs, state, reward, done, _ = eval_env.step(
                        ks, state, list(action))
                    reward = jnp.asarray(reward).reshape(-1)[:n_agents]
                    carry = (state, obs, hstate, action, reward,
                             jnp.full(n_agents, done["__all__"]), key)
                    return carry, reward.mean()

                hstate = ScannedRNN.initialize_carry(
                    n_agents, config["RNN_HIDDEN_SIZE"])
                carry = (state, obs, hstate, jnp.zeros(n_agents, dtype=jnp.int32),
                         jnp.zeros(n_agents), jnp.zeros(n_agents, dtype=bool), key)
                _, rewards = jax.lax.scan(step_fn, carry, None, ep_steps)
                return rewards.sum()

            returns = jax.vmap(one_episode)(
                jax.random.split(key, config.get("EVAL_NUM_ENVS", 64)))
            return jnp.stack([
                returns.mean(),
                jnp.median(returns),
                jnp.percentile(returns, 20),
            ])

        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                (train_state, env_state, last_obs, last_done, prev_action,
                 prev_reward, hstate, update_step, last_eval, rng) = runner_state

                obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(
                    -1, *env.observation_space()[0].shape
                )
                # one timestep: add a leading time axis of length 1
                ac_in = (
                    obs_batch[None, :],
                    prev_action[None, :],
                    prev_reward[None, :],
                    last_done[None, :],
                )
                new_hstate, pi, value, *_ = network.apply(
                    train_state.params, hstate, ac_in)
                rng, _rng = jax.random.split(rng)
                action = pi.sample(seed=_rng).squeeze(0)
                log_prob = pi.log_prob(action[None, :]).squeeze(0)
                value = value.squeeze(0)

                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = [v for v in env_act.values()]

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify_dict(
                    done, env.agents, config["NUM_ACTORS"]
                ).squeeze()
                reward_batch = batchify(
                    reward, env.agents, config["NUM_ACTORS"]
                ).squeeze()

                # No reward shaping. Paying an agent the moment its beam
                # clears a cell would hand it the very signal the experiment
                # asks it to infer ("which tool works here?"), so the policy
                # is trained on the environment's own reward only.

                transition = RecurrentTransition(
                    done_batch,      # after this step -> GAE bootstrapping
                    last_done,       # before this step -> RNN reset signal
                    action,
                    value,
                    reward_batch,
                    log_prob,
                    obs_batch,
                    prev_action,
                    prev_reward,
                    info,
                )
                runner_state = (train_state, env_state, obsv, done_batch,
                                action, reward_batch, new_hstate, update_step,
                                last_eval, rng)
                return runner_state, transition

            # the hidden state at the start of the rollout is what the update
            # has to replay from, so keep a copy before the scan overwrites it
            initial_hstate = runner_state[6]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            (train_state, env_state, last_obs, last_done, prev_action,
             prev_reward, hstate, update_step, last_eval, rng) = runner_state
            last_obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(
                -1, *env.observation_space()[0].shape
            )
            _, _, last_val, *_ = network.apply(
                train_state.params,
                hstate,
                (last_obs_batch[None, :], prev_action[None, :],
                 prev_reward[None, :], last_done[None, :]),
            )
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done, transition.value, transition.reward
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hs, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hs, traj_batch, gae, targets):
                        # replay the whole sequence through the GRU, starting
                        # from the hidden state this minibatch began with
                        _, pi, value, *aux_out = network.apply(
                            params,
                            init_hs,
                            (traj_batch.obs, traj_batch.prev_action,
                             traj_batch.prev_reward, traj_batch.last_done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"],
                                     1.0 + config["CLIP_EPS"]) * gae,
                        ).mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )

                        # Self-supervised auxiliary loss: from the hidden
                        # state at step t, predict whether this step's
                        # cleaning shot clears its target. Masked to steps
                        # where the agent actually fired, so the head is
                        # graded only on questions the rule answers. Probes
                        # showed PPO alone never writes this evidence into
                        # the GRU; this makes doing so part of the objective.
                        if use_aux:
                            aux_logit = aux_out[0]
                            fired = traj_batch.info["clean_action_info"].astype(jnp.float32)
                            label = (traj_batch.info["cleaned_by_agent"] > 0).astype(jnp.float32)
                            bce = optax.sigmoid_binary_cross_entropy(aux_logit, label)
                            aux_loss = (bce * fired).sum() / jnp.clip(fired.sum(), 1.0)
                            total_loss = total_loss + config["AUX_COEF"] * aux_loss
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hs, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                # Split the ACTOR axis and keep time intact: each minibatch is
                # a set of whole sequences the GRU can be replayed over.
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                minibatch_hstate, minibatches = split_sequence_minibatches(
                    permutation,
                    init_hstate,
                    (traj_batch, advantages, targets),
                    config["NUM_MINIBATCHES"],
                    config["NUM_ACTORS_PER_MINIBATCH"],
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state,
                    (minibatch_hstate,) + minibatches
                )
                update_state = (train_state, init_hstate, traj_batch,
                                advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, initial_hstate, traj_batch,
                            advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # Held-out evaluation on a fixed seed, every EVAL_EVERY updates.
            # The result is carried forward between evaluations so the metric
            # is defined on every step of the curve.
            eval_every = config.get("EVAL_EVERY", 50)
            eval_stats = jax.lax.cond(
                (update_step % eval_every) == 0,
                lambda: _eval_on_held_out(
                    train_state.params,
                    jax.random.PRNGKey(config.get("EVAL_SEED", 42))),
                lambda: last_eval,
            )

            # Reduce to per-update means BEFORE the outer scan stacks metrics:
            # raw info leaves are (NUM_STEPS, NUM_ACTORS), and stacking them
            # across NUM_UPDATES allocates TOTAL_TIMESTEPS * num_agents-sized
            # buffers — a 3e8-step run OOMs on exactly that (4.2GB for the
            # first int16 key alone) before the first update runs. The wandb
            # callback and _save_metrics only ever consumed the mean anyway.
            metric = jax.tree.map(
                lambda x: x.astype(jnp.float32).mean(), traj_batch.info
            )
            metric["eval/returns_mean"] = eval_stats[0]
            metric["eval/returns_median"] = eval_stats[1]
            metric["eval/returns_20percentile"] = eval_stats[2]
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            update_step = update_step + 1

            if config.get("WANDB_MODE", "online") != "disabled":
                def callback(metric):
                    wandb.log(
                        jax.tree.map(
                            lambda x: x.mean() if hasattr(x, "mean") else x, metric
                        )
                    )
                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, last_done,
                            prev_action, prev_reward, hstate, update_step,
                            eval_stats, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, init_done,
                        init_prev_action, init_prev_reward, init_hstate, 0,
                        jnp.zeros(3), _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


# Used by algorithms/train.py to dispatch through algorithms.RPPO._runner.
SINGLE_RUN_KWARGS = {"wandb_name": "rppo_cnn_open_cleanup"}
TUNE_KWARGS = {"sweep_name": "open_cleanup"}
