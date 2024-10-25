import time
import jax
import flax.serialization
from luxai_s3.params import EnvParams
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import env_params_ranges
from luxai_s3.state import gen_map
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

if __name__ == "__main__":
    import numpy as np

    np.random.seed(2)

    # the first env params is not batched and is used to initialize any static / unchaging values
    # like map size, max units etc.
    jax_env = LuxAIS3Env(auto_reset=True, fixed_env_params=EnvParams())
    num_envs = 10
    seed = 0
    rng_key = jax.random.key(seed)
    reset_fn = jax.vmap(jax_env.reset_env)

    # sample random params initially
    def sample_params(rng_key):
        randomized_game_params = dict()
        for k, v in env_params_ranges.items():
            rng_key, subkey = jax.random.split(rng_key)
            randomized_game_params[k] = jax.random.choice(subkey, jax.numpy.array(v))
        params = EnvParams(**randomized_game_params)
        return params

    rng_key, subkey = jax.random.split(rng_key)
    env_params = jax.vmap(sample_params)(jax.random.split(subkey, num_envs))
    #
    # env_params = EnvParams()
    # for k in env_params_ranges:
    #     env_params = env_params.replace(**{k: getattr(sampled_env_params, k)})
    # env_params = [EnvParams() for _ in range(num_envs)]
    res = reset_fn(jax.random.split(subkey, num_envs), env_params)

    # def gen_map(key, params):
    #     # jax.debug.breakpoint()
    #     # import ipdb; ipdb.set_trace()
    #     jax.numpy.zeros((params.map_height, 24))
    #     return params.map_height+params.map_width
    # gen_map = jax.jit(gen_map, static_argnums=(0,2,3,4,5,6, 7))
    # import ipdb; ipdb.set_trace()
    # res = jax.vmap(gen_map, in_axes=(0, 0, None, None, None, None, None, None))(jax.random.split(subkey, num_envs), env_params, fixed_env_params.map_type, fixed_env_params.map_height, fixed_env_params.map_width, fixed_env_params.max_energy_nodes, fixed_env_params.max_relic_nodes, fixed_env_params.relic_config_size)

    # env = LuxAIS3GymEnv()
    # env = RecordEpisode(env, save_dir="episodes")
    # obs, info = env.reset(seed=1)

    print("Benchmarking time")
    stime = time.time()
    N = 100
    N = fixed_env_params.max_steps_in_match * fixed_env_params.match_count_per_episode
    for _ in range(N):
        env.step(env.action_space.sample())
    etime = time.time()
    print(f"FPS: {N / (etime - stime)}")

    # env.close()
