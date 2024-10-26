from dataclasses import dataclass
from functools import partial
from typing import Annotated
import jax
import jax.numpy as jnp
import tyro
from luxai_s3.params import EnvParams
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import env_params_ranges
from luxai_s3.profiler import Profiler

@dataclass
class Args:
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 64
    trials_per_benchmark: Annotated[int, tyro.conf.arg(aliases=["-t"])] = 5
    verbose: Annotated[int, tyro.conf.arg(aliases=["-v"])] = 0
    seed: int = 0

if __name__ == "__main__":
    import numpy as np
    jax.config.update('jax_numpy_dtype_promotion', 'strict')
    args = tyro.cli(Args)

    np.random.seed(args.seed)

    # the first env params is not batched and is used to initialize any static / unchaging values
    # like map size, max units etc.
    # note auto_reset=False for speed reasons. If True, the default jax code will attempt to reset each time and discard the reset if its not time to reset
    # due to jax branching logic. It should be kept false and instead lax.scan followed by a reset after max episode steps should be used when possible since games
    # can't end early.
    env = LuxAIS3Env(auto_reset=False, fixed_env_params=EnvParams())
    num_envs = args.num_envs
    seed = args.seed
    rng_key = jax.random.key(seed)
    reset_fn = jax.vmap(env.reset)
    step_fn = jax.vmap(env.step)

    # sample random params initially
    def sample_params(rng_key):
        randomized_game_params = dict()
        for k, v in env_params_ranges.items():
            rng_key, subkey = jax.random.split(rng_key)
            if isinstance(v[0], int):
                randomized_game_params[k] = jax.random.choice(subkey, jax.numpy.array(v, dtype=jnp.int16))
            else:
                randomized_game_params[k] = jax.random.choice(subkey, jax.numpy.array(v, dtype=jnp.float32))
        params = EnvParams(**randomized_game_params)
        return params

    rng_key, subkey = jax.random.split(rng_key)
    env_params = jax.vmap(sample_params)(jax.random.split(subkey, num_envs))
    action_space = env.action_space() # note that this can generate sap actions beyond range atm
    sample_action = jax.vmap(action_space.sample)
    obs, state = reset_fn(jax.random.split(subkey, num_envs), env_params)
    obs, state, reward, terminated_dict, truncated_dict, info = step_fn(
        jax.random.split(subkey, num_envs), 
        state, 
        sample_action(jax.random.split(subkey, num_envs)), 
        env_params
    )

    max_episode_steps = (env.fixed_env_params.max_steps_in_match + 1) * env.fixed_env_params.match_count_per_episode
    rng_key, subkey = jax.random.split(rng_key)
    profiler = Profiler(output_format="stdout")

    
    def benchmark_reset_for_loop_jax_step(rng_key):
        rng_key, subkey = jax.random.split(rng_key)
        states = []
        obs, state = reset_fn(jax.random.split(subkey, num_envs), env_params)
        states.append(state)
        for _ in range(max_episode_steps):
            rng_key, subkey = jax.random.split(rng_key)
            obs, state, reward, terminated_dict, truncated_dict, info = step_fn(
            jax.random.split(subkey, num_envs), 
            state, 
            sample_action(jax.random.split(subkey, num_envs)), 
                env_params
            )
            jax.block_until_ready(state)
            states.append(state)
    profiler.profile(partial(benchmark_reset_for_loop_jax_step, rng_key), "reset + for loop jax.step", total_steps=max_episode_steps, num_envs=num_envs, trials=args.trials_per_benchmark)
    profiler.log_stats("reset + for loop jax.step")


    def run_episode(rng_key, state, env_params):
        def take_step(carry, _):
            rng_key, state = carry
            rng_key, subkey = jax.random.split(rng_key)
            obs, state, reward, terminated_dict, truncated_dict, info = step_fn(
                jax.random.split(subkey, num_envs), 
                state, 
                sample_action(jax.random.split(subkey, num_envs)), 
                env_params
            )
            return (rng_key, state), (obs, state, reward, terminated_dict, truncated_dict, info)
        _, (obs, state, reward, terminated_dict, truncated_dict, info) = jax.lax.scan(take_step, (rng_key, state), length=max_episode_steps, unroll=1)
        return obs, state, reward, terminated_dict, truncated_dict, info
    # compile the scan
    if args.verbose: print("Compiling run_episode")
    run_episode = jax.jit(run_episode)
    run_episode(subkey, state, env_params)
    if args.verbose: print("Compiling run_episode done")
    
    def benchmark_reset_jax_lax_scan_jax_step(rng_key):
        rng_key, subkey = jax.random.split(rng_key)
        obs, state = reset_fn(jax.random.split(subkey, num_envs), env_params)
        rng_key, subkey = jax.random.split(rng_key)
        # obs now has shape (max_episode_steps, num_envs, ...)
        obs, state, reward, terminated_dict, truncated_dict, info = run_episode(subkey, state, env_params)
        jax.block_until_ready(state)
    profiler.profile(partial(benchmark_reset_jax_lax_scan_jax_step, rng_key), "reset + jax.lax.scan(jax.step)", total_steps=max_episode_steps, num_envs=num_envs, trials=args.trials_per_benchmark)
    profiler.log_stats("reset + jax.lax.scan(jax.step)")

    def run_episode_and_reset(rng_key, env_params):
        rng_key, subkey = jax.random.split(rng_key)
        obs, state = reset_fn(jax.random.split(subkey, num_envs), env_params)
        def take_step(carry, _):
            rng_key, state = carry
            rng_key, subkey = jax.random.split(rng_key)
            obs, state, reward, terminated_dict, truncated_dict, info = step_fn(
                jax.random.split(subkey, num_envs), 
                state, 
                sample_action(jax.random.split(subkey, num_envs)), 
                env_params
            )
            return (rng_key, state), (obs, state, reward, terminated_dict, truncated_dict, info)
        _, (obs, state, reward, terminated_dict, truncated_dict, info) = jax.lax.scan(take_step, (rng_key, state), length=max_episode_steps)
        return obs, state, reward, terminated_dict, truncated_dict, info
    # compile the scan
    if args.verbose: print("Compiling run_episode_and_reset")
    run_episode_and_reset = jax.jit(run_episode_and_reset)
    run_episode_and_reset(subkey, env_params)
    if args.verbose: print("Compiling run_episode_and_reset done")
    def benchmark_jit_reset_lax_scan_jax_step(rng_key):
        rng_key, subkey = jax.random.split(rng_key)
        obs, state, reward, terminated_dict, truncated_dict, info = run_episode_and_reset(subkey, env_params)
        jax.block_until_ready(state)
    profiler.profile(partial(benchmark_jit_reset_lax_scan_jax_step, rng_key), "jit(reset + jax.lax.scan(jax.step))", total_steps=max_episode_steps, num_envs=num_envs, trials=args.trials_per_benchmark)
    profiler.log_stats("jit(reset + jax.lax.scan(jax.step))")
