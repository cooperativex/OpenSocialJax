"""RL^2: recurrent PPO for the hidden-rule environments — the training loop shared by every per-env module.

Feed-forward PPO cannot solve a task whose rule is hidden and resampled per meta-episode: with no memory it can only
learn a fixed prior over rules. This is the RL^2 setup (Duan et al., 2016; Wang et al., 2016) that XLand-MiniGrid
uses for its meta-tasks — a recurrent actor-critic that also sees the previous action and previous reward, so "I tried
that and it did not pay" is available as evidence within an episode.

Multi-agent: every agent of an environment acts at once. One recurrent policy drives all of them (parameter
sharing), each with its own hidden state, observation and reward, so an environment with N agents contributes N
actors. That is the usual RL^2 multi-agent setup and keeps the sequence bookkeeping tractable.

Two things differ from the feed-forward IPPO loop, and both matter:

1. The hidden state is carried through the rollout and reset on episode boundaries. In open_cleanup and open_harvest
   `done` only fires at the end of a meta-episode, so memory persists across the trials that share a rule (and a
   map) — which is the whole point.
2. Minibatching splits the **actor** axis only and keeps the time axis intact, because the update replays each
   sequence through the GRU. The feed-forward loop flattens (time x actor) and shuffles, which would destroy the
   sequences.

Rewards (config REWARD): the environments pay each agent its own reward, and that is what `individual` trains on.
`common` trains every agent on the sum over the agents of its environment (what SocialJax's shared_rewards did).
The switch lives here, not in the environments, so the environments and the LLM harness stay individual-only. The
previous reward the policy sees is the one it is trained on; returns are always logged as the environment's own,
per agent (with common rewards the team's total is that times the number of agents).

Actor order: batchify lays actors out agent-major (index = agent * NUM_ENVS + env); every per-actor array here —
observations, actions, rewards, hidden states and the info leaves — follows that order.
"""
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import wandb

from opensocialjax.wrappers.baselines import LogWrapper
from algorithms.utils import (
    ActorCriticRNN,
    ScannedRNN,
    batchify,
    batchify_dict,
    unbatchify,
    load_params,
    RecurrentTransition,
)

REWARDS = ("individual", "common")


def training_reward(reward, mode):
    """`reward`: (..., num_agents) of the environment's own rewards. `individual` keeps them; `common` gives every
    agent the sum over the agents of its environment."""
    assert mode in REWARDS, f"REWARD must be one of {REWARDS}, got {mode!r}"
    if mode == "individual":
        return reward
    return jnp.broadcast_to(reward.sum(axis=-1, keepdims=True), reward.shape)


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


def make_network(config, env, obs_layout):
    return ActorCriticRNN(
        env.action_space().n,
        activation=config["ACTIVATION"],
        hidden_size=config["RNN_HIDDEN_SIZE"],
        head_hidden_size=config.get("HEAD_HIDDEN_SIZE", 64),
        obs_layout=tuple(obs_layout),
        aux=float(config.get("AUX_COEF", 0.0)) > 0.0,
    )


def make_train(config, env, eval_env, obs_layout, aux_target=None):
    """The RL^2 training function for one environment.

    env / eval_env: the training and the held-out environment (unwrapped; the same game on disjoint rule and map
    splits). obs_layout: the (kind, colour, agent-feature, held-tool) channel counts of the observation, see
    ActorCriticRNN. aux_target: optional `info -> (label, mask)`, both (T, actors), for the AUX_COEF loss: from the
    hidden state at step t predict `label`, graded only where `mask` is on.
    """
    reward_mode = config.get("REWARD", "individual")
    assert reward_mode in REWARDS, f"REWARD must be one of {REWARDS}, got {reward_mode!r}"
    if not config.get("PARAMETER_SHARING", True):
        raise ValueError(
            "RPPO implements the shared-policy RL^2 setup; set "
            "PARAMETER_SHARING=True (per-agent recurrent policies would need "
            "separate hidden-state bookkeeping)."
        )
    use_aux = float(config.get("AUX_COEF", 0.0)) > 0.0
    assert not use_aux or aux_target is not None, "AUX_COEF > 0 but this environment defines no auxiliary target"

    n_agents = env.num_agents
    config["NUM_ACTORS"] = n_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
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
    inner, outer = int(eval_env.num_inner_steps), int(eval_env.num_outer_steps)
    obs_shape = env.observation_space()[0].shape

    env = LogWrapper(env, replace_info=False)

    def per_actor(info):
        """Info leaves to one value per actor, agent-major like the actors: (NUM_ENVS, n_agents) leaves are
        transposed and flattened, per-environment (NUM_ENVS,) leaves are given to each of its agents, anything else
        (per-cell diagnostics) is dropped."""
        out, dropped = {}, []
        for k, v in info.items():
            if v.shape == (config["NUM_ENVS"], n_agents):
                out[k] = v.T.reshape(config["NUM_ACTORS"])
            elif v.shape == (config["NUM_ENVS"],):
                out[k] = jnp.tile(v, n_agents)
            else:
                dropped.append(k)
        if not per_actor.reported:      # once, at trace time
            print(f"[rl2] info logged per actor: {sorted(out)}; dropped: {sorted(dropped)}")
            per_actor.reported = True
        return out
    per_actor.reported = False

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
                      ) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        network = make_network(config, env, obs_layout)
        rng, _rng = jax.random.split(rng)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["RNN_HIDDEN_SIZE"]
        )
        init_x = (
            jnp.zeros((1, config["NUM_ACTORS"], *obs_shape)),
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
            """Roll out whole meta-episodes on held-out rules and maps.

            Returns the mean, median and 20th percentile of the meta-episode
            return (per agent) — XLand logs the 20th percentile because a
            meta-policy that solves the easy rules and fails the rest has a
            respectable mean and a terrible tail — followed by the mean return
            of each trial. Within-meta-episode adaptation shows as later
            trials scoring more than the first."""
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
                    reward = jnp.asarray(reward, dtype=jnp.float32).reshape(-1)[:n_agents]
                    carry = (state, obs, hstate, action,
                             training_reward(reward, reward_mode),
                             jnp.full(n_agents, done["__all__"]), key)
                    return carry, reward.mean()

                hstate = ScannedRNN.initialize_carry(
                    n_agents, config["RNN_HIDDEN_SIZE"])
                carry = (state, obs, hstate, jnp.zeros(n_agents, dtype=jnp.int32),
                         jnp.zeros(n_agents), jnp.zeros(n_agents, dtype=bool), key)
                _, rewards = jax.lax.scan(step_fn, carry, None, inner * outer)
                return rewards.reshape(outer, inner).sum(axis=1)

            per_trial = jax.vmap(one_episode)(
                jax.random.split(key, config.get("EVAL_NUM_ENVS", 64)))
            returns = per_trial.sum(axis=1)
            return jnp.concatenate([
                jnp.stack([returns.mean(), jnp.median(returns),
                           jnp.percentile(returns, 20)]),
                per_trial.mean(axis=0),
            ])

        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                (train_state, env_state, last_obs, last_done, prev_action,
                 prev_reward, hstate, update_step, last_eval, rng) = runner_state

                obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(
                    -1, *obs_shape
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
                # one (NUM_ENVS,) array per agent, so every environment gets one scalar action per agent (a
                # trailing axis of 1 would broadcast against the environments' per-agent masks)
                env_act = [v.squeeze(-1) for v in env_act.values()]

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                info = per_actor(info)
                done_batch = batchify_dict(
                    done, env.agents, config["NUM_ACTORS"]
                ).squeeze()
                # (NUM_ENVS, n_agents): the reward this run trains on, then one per actor
                reward = training_reward(
                    jnp.asarray(reward, dtype=jnp.float32).reshape(config["NUM_ENVS"], n_agents), reward_mode)
                reward_batch = batchify(
                    reward, env.agents, config["NUM_ACTORS"]
                ).squeeze()

                # No reward shaping. Paying an agent for a step towards the
                # rule (a cleared cell, a well-timed apple) would hand it the
                # very signal the experiment asks it to infer, so the policy
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
                -1, *obs_shape
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
                        # state at step t, predict the environment's label
                        # for that step (open_cleanup: does this cleaning
                        # shot clear its target), graded only where the mask
                        # is on. Probes showed PPO alone never writes this
                        # evidence into the GRU; this makes doing so part of
                        # the objective.
                        if use_aux:
                            label, mask = aux_target(traj_batch.info)
                            bce = optax.sigmoid_binary_cross_entropy(aux_out[0], label)
                            aux_loss = (bce * mask).sum() / jnp.clip(mask.sum(), 1.0)
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
            total_loss, (value_loss, actor_loss, entropy) = jax.tree.map(jnp.mean, loss_info)

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
            metric["train/reward_per_step"] = traj_batch.reward.mean()
            metric["loss/total"] = total_loss
            metric["loss/value"] = value_loss
            metric["loss/actor"] = actor_loss
            metric["loss/entropy"] = entropy
            metric["eval/returns_mean"] = eval_stats[0]
            metric["eval/returns_median"] = eval_stats[1]
            metric["eval/returns_20percentile"] = eval_stats[2]
            for k in range(outer):
                metric[f"eval/return_trial_{k + 1}"] = eval_stats[3 + k]
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
                        jnp.zeros(3 + outer), _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train
