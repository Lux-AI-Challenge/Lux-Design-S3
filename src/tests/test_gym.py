import time

import flax.serialization
from luxai_s3.params import EnvParams

from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

if __name__ == "__main__":
    import numpy as np

    np.random.seed(2)
    env = LuxAIS3GymEnv()
    env = RecordEpisode(env, save_dir="episodes")
    env_params = EnvParams(map_type=0, max_steps_in_match=100)
    obs, info = env.reset(seed=1, options=dict(params=env_params))

    print("Benchmarking time")
    stime = time.time()
    N = env_params.max_steps_in_match * env_params.match_count_per_episode
    for _ in range(N):
        env.step(env.action_space.sample())
    etime = time.time()
    print(f"FPS: {N / (etime - stime)}")

    env.close()
