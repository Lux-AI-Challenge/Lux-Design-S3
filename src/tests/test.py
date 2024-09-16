import json
import time

import flax
from luxai_s3.params import EnvParams
from luxai_s3.state import EnvState, env_states_to_dict

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    from luxai_s3.env import LuxAIS3Env
    # from luxai_s3.wrappers import RecordEpisode

    # Create the environment
    env = LuxAIS3Env(auto_reset=False)
    env_params = EnvParams(map_type="dev0")

    # Initialize a random key
    key = jax.random.PRNGKey(0)

    # Reset the environment
    key, subkey = jax.random.split(key)
    obs, state = env.reset(subkey, params=env_params)
    # Take a random action
    key, subkey = jax.random.split(key)
    action = env.action_space(env_params).sample(subkey)
    # Step the environment
    key, subkey = jax.random.split(key)
    obs, state, reward, terminated, truncated, info = env.step(
        subkey, state, action, params=env_params
    )



    states = []
    key, subkey = jax.random.split(key)
    obs, state = env.reset(subkey, params=env_params)
    states.append(state)
    print("Benchmarking time")
    stime = time.time()
    N = 1000
    for _ in range(N):
        key, subkey = jax.random.split(key)
        action = env.action_space(env_params).sample(subkey)
        obs, state, reward, terminated, truncated, info = env.step(
            subkey, state, action, params=env_params
        )
        states.append(state)
        env.render(state, env_params)
    etime = time.time()
    print(f"FPS: {N / (etime - stime)}")
    episode = env_states_to_dict(states)
    episode["params"] = flax.serialization.to_state_dict(env_params)
    with open("episode.json", "w") as f:
        json.dump(episode, f, indent=4)
    import ipdb; ipdb.set_trace()