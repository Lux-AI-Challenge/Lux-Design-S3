import functools
from typing import Any, Dict, Optional, Tuple, Union

import chex
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces
from jax import lax

from luxai_s3.params import EnvParams
from luxai_s3.spaces import MultiDiscrete
from luxai_s3.state import EnvObs, EnvState, gen_state
from luxai_s3.pygame_render import LuxAIPygameRenderer


class LuxAIS3Env(environment.Environment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.renderer = LuxAIPygameRenderer()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[EnvObs, EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        # state = state.replace() # TODO (stao)
        action = jnp.stack([action["team_0"], action["team_1"]])
        ### process unit movement ###
        # 0 is do nothing, 1 is move up, 2 is move right, 3 is move down, 4 is move left
        # Define movement directions
        directions = jnp.array(
            [
                [0, 0],  # Do nothing
                [0, -1],  # Move up
                [1, 0],  # Move right
                [0, 1],  # Move down
                [-1, 0],  # Move left
            ],
            dtype=jnp.int16,
        )

        def move_unit(unit, action, mask):
            new_pos = unit[:2] + directions[action]
            # Ensure the new position is within the map boundaries
            new_pos = jnp.clip(
                new_pos,
                0,
                jnp.array(
                    [params.map_width - 1, params.map_height - 1], dtype=jnp.int16
                ),
            )
            # Update the unit's position only if it's active
            return jnp.where(mask, jnp.concatenate([new_pos, unit[2:]]), unit)

        # Move units for both teams
        for team in range(2):
            state = state.replace(
                units=state.units.at[team].set(
                    jax.vmap(move_unit)(
                        state.units[team], action[team], state.units_mask[team]
                    )
                )
            )

        # Update state's step count
        state = state.replace(steps=state.steps + 1)

        reward = jnp.array(0.0)
        terminated = self.is_terminal(state, params)
        truncated = state.steps >= params.max_steps_in_episode
        return (
            lax.stop_gradient(self.get_obs(state, params, key=key)),
            lax.stop_gradient(state),
            reward,
            terminated,
            truncated,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[EnvObs, EnvState]:
        """Reset environment state by sampling initial position."""

        state = gen_state(key=key, params=params)

        return self.get_obs(state, params=params, key=key), state

    # @functools.partial(jax.jit, static_argnums=(0,4))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: Optional[EnvParams] = None,
    ) -> Tuple[EnvObs, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)

        obs_st, state_st, reward, terminated, truncated, info = self.step_env(
            key, state, action, params
        )
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on done
        done = terminated | truncated
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        info["final_observation"] = obs_st
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, terminated, truncated, info

    # @functools.partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        obs, state = self.reset_env(key, params)
        return obs, state

    # @functools.partial(jax.jit, static_argnums=(0, 2))
    def get_obs(self, state: EnvState, params=None, key=None) -> EnvObs:
        """Return observation from raw state, handling partial observability."""
        obs = jnp.zeros(shape=(3, 3, 2), dtype=jnp.float32)
        # if params.fog_of_war:
        #     pass
        # else:
        #     obs = state
        return obs

    @functools.partial(jax.jit, static_argnums=(0, 2))
    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal. This occurs when either team wins/loses outright."""
        terminated = jnp.array(False)
        return terminated

    @property
    def name(self) -> str:
        """Environment name."""
        return "Lux AI Season 3"

    def render(self, state: EnvState, params: EnvParams):
        self.renderer.render(state, params)

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        size = np.ones(params.max_units) * 5
        return spaces.Dict(dict(team_0=MultiDiscrete(size), team_1=MultiDiscrete(size)))

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return spaces.Discrete(10)

    def state_space(self, params: EnvParams):
        """State space of the environment."""
        return spaces.Discrete(10)
