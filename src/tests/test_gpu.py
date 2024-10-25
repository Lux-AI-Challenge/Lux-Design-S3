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
    
    jax_env = LuxAIS3Env(auto_reset=True)
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
    # reset_fn(jax.random.split(subkey, num_envs), env_params)
    jax.vmap(gen_map)(jax.random.split(subkey, num_envs), env_params)


    
    # env = LuxAIS3GymEnv()
    # env = RecordEpisode(env, save_dir="episodes")
    # obs, info = env.reset(seed=1)
    
    # print("Benchmarking time")
    # stime = time.time()
    # N = 100
    # # N = env.params.max_steps_in_match * env.params.match_count_per_episode
    # for _ in range(N):
    #     env.step(env.action_space.sample())
    # etime = time.time()
    # print(f"FPS: {N / (etime - stime)}")
    
    # env.close()