"""Random-action throughput of the environments, after JaxMARL's speed test.

Steps NUM_ENVS copies of an environment in lockstep for NUM_STEPS steps under jax.vmap + jax.lax.scan, with uniformly
random actions, and reports environment steps per second on the default JAX device (the GPU when there is one).
Compilation is excluded from the timing.

    python speed_test/speed_test_random.py                                     # both environments, 1 / 128 / 1024 copies
    python speed_test/speed_test_random.py --env open_harvest --num-envs 4096 --num-steps 1000
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax

import opensocialjax
from opensocialjax.registration import REGISTERED_ENVS


def make_benchmark(env, num_envs, num_steps):
    def benchmark(rng):
        rng, r_reset = jax.random.split(rng)
        _, state = jax.vmap(env.reset)(jax.random.split(r_reset, num_envs))

        def step(carry, _):
            state, rng = carry
            rng, r_act, r_step = jax.random.split(rng, 3)
            r_act = jax.random.split(r_act, env.num_agents * num_envs).reshape(env.num_agents, num_envs, -1)
            actions = [jax.vmap(env.action_space(a).sample)(r_act[i]) for i, a in enumerate(env.agents)]
            _, state, _, _, _ = jax.vmap(env.step)(jax.random.split(r_step, num_envs), state, actions)
            return (state, rng), None

        (state, _), _ = jax.lax.scan(step, (state, rng), None, num_steps)
        return state

    return benchmark


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="all", choices=["all"] + REGISTERED_ENVS)
    ap.add_argument("--num-envs", default="1,128,1024", help="parallel copies, comma-separated")
    ap.add_argument("--num-steps", type=int, default=1000)
    ap.add_argument("--agents", type=int, default=5)
    a = ap.parse_args()

    dev = jax.devices()[0]
    print(f"device: {dev.platform} ({dev.device_kind})")
    for env_id in (REGISTERED_ENVS if a.env == "all" else [a.env]):
        env = opensocialjax.make(env_id, num_agents=a.agents)
        for n in [int(x) for x in a.num_envs.split(",")]:
            fn = jax.jit(make_benchmark(env, n, a.num_steps)).lower(jax.random.PRNGKey(0)).compile()
            t0 = time.perf_counter()
            jax.block_until_ready(fn(jax.random.PRNGKey(0)))
            dt = time.perf_counter() - t0
            steps = n * a.num_steps
            print(f"{env_id:13s} {a.agents} agents  num_envs {n:5d}  {a.num_steps} steps  {dt:7.2f} s  "
                  f"{steps / dt:12,.0f} env steps/s  {steps * a.agents / dt:12,.0f} agent steps/s")


if __name__ == "__main__":
    main()
