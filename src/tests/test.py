import time
from luxai_s3.params import EnvParams

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

    states = []
    # Reset the environment
    key, subkey = jax.random.split(key)
    obs, state = env.reset(subkey, params=env_params)
    states.append(state)
    # Take a random action
    key, subkey = jax.random.split(key)
    action = env.action_space(env_params).sample(subkey)
    # Step the environment
    key, subkey = jax.random.split(key)
    obs, state, reward, terminated, truncated, info = env.step(
        subkey, state, action, params=env_params
    )
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
