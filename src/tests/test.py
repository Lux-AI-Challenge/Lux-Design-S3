from luxai_s3.params import EnvParams


if __name__ == "__main__":
    from luxai_s3.env import LuxAIS3Env
    import jax
    import jax.numpy as jnp

    # Create the environment
    env = LuxAIS3Env()
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
    obs, state, reward, terminated, truncated, info = env.step(subkey, state, action, params=env_params)

    print("Observation:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)

    while True:
        key, subkey = jax.random.split(key)
        action = env.action_space(env_params).sample(subkey)
        # import ipdb;ipdb.set_trace()
        obs, state, reward, terminated, truncated, info = env.step(subkey, state, action, params=env_params)
        # env.render(state, env_params)