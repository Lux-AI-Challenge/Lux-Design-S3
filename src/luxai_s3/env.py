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
    def __init__(self, auto_reset=False, **kwargs):
        super().__init__(**kwargs)
        self.renderer = LuxAIPygameRenderer()
        self.auto_reset = auto_reset

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
            # Check if the new position is on a map feature of value 2
            is_blocked = state.map_features[new_pos[1], new_pos[0], 0] == 2
            # If blocked, keep the original position
            new_pos = jnp.where(is_blocked, unit[:2], new_pos)
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
        state = state.replace(
            units=jax.vmap(lambda team_units, team_action, team_mask: jax.vmap(move_unit)(team_units, team_action, team_mask))(
                state.units, action, state.units_mask
            )
        )

        # compute the sensor mask for both teams
        sensor_mask = jnp.zeros(shape=(params.num_teams, params.map_height, params.map_width), dtype=jnp.bool_)
        # sensor_mask = sensor_mask.at[state.units[0][1], state.units[0][0]].set(True)
        # sensor_mask = sensor_mask.at[state.units[1][1], state.units[1][0]].set(True)
        # Update sensor mask based on the sensor range
        # def update_sensor_mask(unit_pos, sensor_mask):
        #     y, x = unit_pos
        #     for dy in range(-params.unit_sensor_range, params.unit_sensor_range + 1):
        #         for dx in range(-params.unit_sensor_range, params.unit_sensor_range + 1):
        #             new_y, new_x = y + dy, x + dx
        #             in_range = dx + dy <= params.unit_sensor_range
        #             in_bounds = (0 <= new_y) & (new_y < params.map_height) & (0 <= new_x) & (new_x < params.map_width)
        #             sensor_mask = jnp.where(
        #                 in_range & in_bounds,
        #                 sensor_mask.at[new_y, new_x].set(True),
        #                 sensor_mask
        #             )
        #     return sensor_mask

        # # Apply the sensor mask update for all units of both teams
        # def update_unit_sensor_mask(unit_pos, mask, sensor_mask):
        #     return jax.lax.cond(
        #         mask,
        #         lambda: update_sensor_mask(unit_pos, sensor_mask),
        #         lambda: sensor_mask
        #     )

        # def update_team_sensor_mask(team_units, team_mask, sensor_mask):
        #     def body_fun(carry, i):
        #         sensor_mask = carry
        #         return update_unit_sensor_mask(team_units[i, :2], team_mask[i], sensor_mask), None

        #     final_sensor_mask, _ = jax.lax.scan(body_fun, sensor_mask, jnp.arange(params.max_units))
        #     return final_sensor_mask

        # sensor_mask = jax.vmap(update_team_sensor_mask, in_axes=(0, 0, 0))(
        #     state.units,
        #     state.units_mask,
        #     sensor_mask
        # )
        for team in range(params.num_teams):
            for unit in state.units[team][state.units_mask[team]]:
                sensor_mask = sensor_mask.at[
                    team, 
                    jnp.maximum(unit[1]-params.unit_sensor_range, 0):unit[1]+params.unit_sensor_range+1, jnp.maximum(unit[0]-params.unit_sensor_range, 0):unit[0]+params.unit_sensor_range+1].set(True)

        state = state.replace(sensor_mask=sensor_mask)

        # Compute relic scores
        def compute_relic_score(unit, relic_nodes_map_weights, mask):
            total_score = relic_nodes_map_weights[unit[1], unit[0]]
            return total_score & mask

        def team_relic_score(units, units_mask):
            scores = jax.vmap(compute_relic_score, in_axes=(0, None, 0))(
                units,
                state.relic_nodes_map_weights,
                units_mask,
            )
            return jnp.sum(scores, dtype=jnp.int32)

        team_0_score = team_relic_score(state.units[0], state.units_mask[0])
        team_1_score = team_relic_score(state.units[1], state.units_mask[1])

        # Update team points
        state = state.replace(
            team_points=state.team_points.at[0].add(team_0_score)
        )
        state = state.replace(
            team_points=state.team_points.at[1].add(team_1_score)
        )
        print(state.team_points)
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
        if self.auto_reset:
            obs_re, state_re = self.reset_env(key_reset, params)
            state = jax.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            info["final_observation"] = obs_st
            obs = jax.lax.select(done, obs_re, obs_st)
        else:
            obs = obs_st
            state = state_st
        # Auto-reset environment based on done
        done = terminated | truncated
        
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
