import json
import time

import flax
import flax.serialization
from luxai_s3.params import EnvParams
from luxai_s3.state import EnvState, serialize_env_actions, serialize_env_states

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    from luxai_s3.env import LuxAIS3Env

    # from luxai_s3.wrappers import RecordEpisode

    # Create the environment
    env = LuxAIS3Env(auto_reset=False)
    env_params = EnvParams(map_type=0, max_steps_in_match=50)

    # Initialize a random key
    key = jax.random.key(0)

    # Reset the environment
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key, params=env_params)
    # Take a random action
    key, subkey = jax.random.split(key)
    action = env.action_space(env_params).sample(subkey)
    # Step the environment and compile. Not sure why 2 steps? are needed
    for _ in range(2):
        key, subkey = jax.random.split(key)
        obs, state, reward, terminated, truncated, info = env.step(
            subkey, state, action, params=env_params
        )

    states = []
    actions = []
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    obs, state = env.reset(subkey, params=env_params)
    states.append(state)
    print("Benchmarking time")
    stime = time.time()
    N = env_params.max_steps_in_match * env_params.match_count_per_episode
    for _ in range(N):
        key, subkey = jax.random.split(key)
        action = env.action_space(env_params).sample(subkey)
        actions.append(action)
        obs, state, reward, terminated, truncated, info = env.step(
            subkey, state, action, params=env_params
        )
        states.append(state)
        # env.render(state, env_params)
    etime = time.time()
    print(f"FPS: {N / (etime - stime)}")

    save_start_time = time.time()

    states = serialize_env_states(states)
    episode = dict(observations=states, actions=serialize_env_actions(actions))
    episode["params"] = flax.serialization.to_state_dict(env_params)
    episode["metadata"] = dict(seed=0)

    # jax.random.PRNGKey(episode["seed"])
    # obs, state = env.reset(jax.random.wrap_key_data(episode["seed"]), params=env_params)
    with open("../lux-eye/src/pages/home/episode.json", "w") as f:
        json.dump(episode, f)
    save_end_time = time.time()

    save_duration = save_end_time - save_start_time
    print(f"Time taken to save episode: {save_duration:.4f} seconds")
    # import ipdb; ipdb.set_trace()
