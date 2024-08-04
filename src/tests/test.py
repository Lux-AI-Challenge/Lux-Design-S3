if __name__ == "__main__":
    from luxai_s3.env import LuxAIS3Env
    import jax
    import jax.numpy as jnp

    # Create the environment
    env = LuxAIS3Env()
    env_params = LuxAIS3Env.default_params

    # Initialize a random key
    key = jax.random.PRNGKey(0)

    # Reset the environment
    key, subkey = jax.random.split(key)
    obs, state = env.reset(subkey)

    # Take a random action
    key, subkey = jax.random.split(key)
    action = env.action_space(env_params).sample(subkey)

    # Step the environment
    key, subkey = jax.random.split(key)
    obs, state, reward, terminated, truncated, info = env.step(subkey, state, action)

    print("Observation:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)