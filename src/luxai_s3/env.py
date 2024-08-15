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
from luxai_s3.state import ASTEROID_TILE, ENERGY_NODE_FNS, NEBULA_TILE, EnvObs, EnvState, UnitState, gen_state
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
        action = jnp.stack([action["team_0"], action["team_1"]]) * 0 + 1
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

        def move_unit(unit: UnitState, action, mask):
            new_pos = unit.position + directions[action]
            # Check if the new position is on a map feature of value 2
            is_blocked = state.map_features.tile_type[new_pos[0], new_pos[1]] == ASTEROID_TILE
            enough_energy = unit.energy >= params.unit_move_cost
            # If blocked, keep the original position
            # new_pos = jnp.where(is_blocked, unit.position, new_pos)
            # Ensure the new position is within the map boundaries
            new_pos = jnp.clip(
                new_pos,
                0,
                jnp.array(
                    [params.map_width - 1, params.map_height - 1], dtype=jnp.int16
                ),
            )
            unit_moved = mask & ~is_blocked & enough_energy
            # Update the unit's position only if it's active
            return UnitState(position=jnp.where(unit_moved, new_pos, unit.position), energy=jnp.where(unit_moved, unit.energy - params.unit_move_cost, unit.energy))

        # Move units for both teams
        # jax.debug.breakpoint()
        # import ipdb; ipdb.set_trace()
        state = state.replace(
            units=jax.vmap(
                lambda team_units, team_action, team_mask: jax.vmap(move_unit, in_axes=(0, 0, 0))(
                    team_units, team_action, team_mask
                ), in_axes=(0, 0, 0)
            )(state.units, action, state.units_mask)
        )

        """Compute energy field of the map and apply it to the units"""
        # first compute a array of shape (map_height, map_width, num_energy_nodes) with values equal to the distance of the tile to the energy node
        mm = jnp.meshgrid(jnp.arange(params.map_width), jnp.arange(params.map_height))
        mm = jnp.stack([mm[0], mm[1]]).T # mm[x, y] gives [x, y]
        distances_to_nodes = jax.vmap(lambda pos: jnp.linalg.norm(mm - pos, axis=-1))(state.energy_nodes)
        def compute_energy_field(node_fn_spec, distances_to_node, mask):
            fn_i, x, y, z = node_fn_spec
            return jnp.where(mask, lax.switch(fn_i, ENERGY_NODE_FNS, distances_to_node, x, y, z), jnp.zeros_like(distances_to_node))
        energy_field = jax.vmap(compute_energy_field)(state.energy_node_fns, distances_to_nodes, state.energy_nodes_mask)
        energy_field = jnp.round(energy_field.sum(0))
        state = state.replace(energy_field=energy_field)

        # Update unit energy based on the energy field of their current position
        def update_unit_energy(unit: UnitState, mask):
            x, y = unit.position
            energy_gain = state.energy_field[x, y]
            new_energy = jnp.clip(unit.energy + energy_gain, params.min_unit_energy, params.max_unit_energy)
            return UnitState(position=unit.position, energy=jnp.where(mask, new_energy, unit.energy))

        # Apply the energy update for all units of both teams
        state = state.replace(
            units=jax.vmap(
                lambda team_units, team_mask: jax.vmap(update_unit_energy)(
                    team_units, team_mask
                )
            )(state.units, state.units_mask)
        )


        """Compute the vision power and sensor mask for both teams 
        
        Algorithm:

        For each team, generate a integer vision power array over the map. 
        For each unit in team, add unit sensor range value (its kind of like the units sensing power/depth) to each tile the unit's sensor range
        Clamp the vision power array to range [0, unit_sensing_range].

        With 2 vision power maps, take the nebula vision mask * nebula vision power and subtract it from the vision power maps.
        Now any time the vision power map has value > 0, the team can sense the tile. This forms the sensor mask
        """

        vision_power_map_padding = params.unit_sensor_range
        vision_power_map = jnp.zeros(
            shape=(params.num_teams, params.map_height + 2 * vision_power_map_padding, params.map_width + 2 * vision_power_map_padding),
            dtype=jnp.int16,
        )

        # Update sensor mask based on the sensor range
        def update_vision_power_map(unit_pos, sensor_mask):
            x, y = unit_pos
            update = jnp.ones(shape= (params.unit_sensor_range * 2 + 1, params.unit_sensor_range * 2 + 1), dtype=jnp.int16)
            for i in range(params.unit_sensor_range + 1):
                update = update.at[i:params.unit_sensor_range * 2 + 1 - i, i:params.unit_sensor_range * 2 + 1 - i].set(i + 1)
            x, y = unit_pos
            sensor_mask = jax.lax.dynamic_update_slice(
                sensor_mask,
                update=update,
                start_indices=(
                    y - params.unit_sensor_range + vision_power_map_padding,
                    x - params.unit_sensor_range + vision_power_map_padding,
                ),
            )
            return sensor_mask

        # Apply the sensor mask update for all units of both teams
        def update_unit_vision_power_map(unit_pos, mask, sensor_mask):
            return jax.lax.cond(
                mask,
                lambda: update_vision_power_map(unit_pos, sensor_mask),
                lambda: sensor_mask,
            )

        def update_team_vision_power_map(team_units, team_mask, sensor_mask):
            def body_fun(carry, i):
                sensor_mask = carry
                return (
                    update_unit_vision_power_map(
                        team_units.position[i], team_mask[i], sensor_mask
                    ),
                    None,
                )

            final_sensor_mask, _ = jax.lax.scan(
                body_fun, sensor_mask, jnp.arange(params.max_units)
            )
            return final_sensor_mask

        vision_power_map = jax.vmap(update_team_vision_power_map)(
            state.units, state.units_mask, vision_power_map
        )
        vision_power_map = vision_power_map[:, vision_power_map_padding:-vision_power_map_padding, vision_power_map_padding:-vision_power_map_padding]

        # handle nebula tiles
        vision_power_map = vision_power_map - (state.map_features.tile_type == NEBULA_TILE)[..., 0] * params.nebula_tile_vision_reduction
        
        sensor_mask = vision_power_map > 0
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

        # team_0_score = team_relic_score(state.units[0], state.units_mask[0])
        # team_1_score = team_relic_score(state.units[1], state.units_mask[1])

        # # Update team points
        # state = state.replace(
        #     team_points=state.team_points.at[0].add(team_0_score)
        # )
        # state = state.replace(
        #     team_points=state.team_points.at[1].add(team_1_score)
        # )
        # print(state.team_points)
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

    @functools.partial(jax.jit, static_argnums=(0,4))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: Optional[EnvParams] = None,
    ) -> Tuple[EnvObs, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        print("Compiled step")
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, terminated, truncated, info = self.step_env(
            key, state, action, params
        )
        if self.auto_reset:
            done = terminated | truncated
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

    @functools.partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        print("Compiled reset")
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
