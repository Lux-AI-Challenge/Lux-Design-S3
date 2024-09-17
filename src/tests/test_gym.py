import json
import time

import flax
import flax.serialization
from luxai_s3.params import EnvParams
from luxai_s3.state import EnvState, serialize_env_actions, serialize_env_states
import jax
import jax.numpy as jnp

from luxai_s3.env import LuxAIS3Env
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
if __name__ == "__main__":
    # env = LuxAIS3Env(auto_reset=False)
    env = LuxAIS3GymEnv()
    env = RecordEpisode(env, save_dir="episodes")
    env_params = EnvParams(map_type=0, max_steps_in_match=100)
    obs, info = env.reset(seed=0, options=dict(params=env_params))
    
    print("Benchmarking time")
    stime = time.time()
    N = env_params.max_steps_in_match * env_params.match_count_per_episode
    for _ in range(N):
        env.step(env.action_space.sample(jax.random.key(0)))
    etime = time.time()
    print(f"FPS: {N / (etime - stime)}")
    
    env.close()

    # save_start_time = time.time()


    # states = serialize_env_states(states)
    # episode=dict(observations=states, actions=serialize_env_actions(actions))
    # episode["params"] = flax.serialization.to_state_dict(env_params)
    # episode["metadata"] = dict(
    #     seed=0
    # )
    
    # # jax.random.PRNGKey(episode["seed"])
    # # obs, state = env.reset(jax.random.wrap_key_data(episode["seed"]), params=env_params)
    # with open("../lux-eye/src/pages/home/episode.json", "w") as f:
    #     json.dump(episode, f)
    # save_end_time = time.time()
    
    # save_duration = save_end_time - save_start_time
    # print(f"Time taken to save episode: {save_duration:.4f} seconds")
    # import ipdb; ipdb.set_trace()