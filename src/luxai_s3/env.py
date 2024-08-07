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


class LuxAIS3Env(environment.Environment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        """Render the environment."""
        import pygame

        tile_size = 64

        # Initialize Pygame if not already done
        if not pygame.get_init():
            pygame.init()

            # Set up the display
            screen_width = params.map_width * tile_size
            screen_height = params.map_height * tile_size
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            self.surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            pygame.display.set_caption("Lux AI Season 3")

            self.display_options = {
                "show_grid": True,
                "show_relic_spots": True,
            }

        # Fill the screen with a background color
        self.screen.fill((200, 200, 200))
        self.surface.fill((200, 200, 200, 255))  # Light gray background

        # Draw the grid of tiles
        for x in range(params.map_width):
            for y in range(params.map_height):
                rect = pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size)
                tile_type = state.map_features[y, x, 0]
                if tile_type == 1:
                    color = (166, 177, 225, 255)  # Light blue (a6b1e1) for tile type 1
                else:
                    color = (255, 255, 255, 255)  # White for other tile types
                pygame.draw.rect(self.surface, color, rect)  # Draw filled squares

        # Draw relic node configs if display option is enabled
        def draw_rect_alpha(surface, color, rect):
            shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
            surface.blit(shape_surf, rect)
        if self.display_options["show_relic_spots"]:
            for i in range(params.max_relic_nodes):
                if state.relic_nodes_mask[i]:
                    x, y = state.relic_nodes[i, :2]
                    config_size = params.relic_config_size
                    half_size = config_size // 2
                    for dx in range(-half_size, half_size + 1):
                        for dy in range(-half_size, half_size + 1):
                            config_x = x + dx
                            config_y = y + dy
                            
                            if (0 <= config_x < params.map_width and 
                                0 <= config_y < params.map_height):
                                
                                config_value = state.relic_node_configs[i, 
                                    dy + half_size, dx + half_size]
                                
                                if config_value > 0 :
                                    rect = pygame.Rect(
                                        config_x * tile_size, 
                                        config_y * tile_size, 
                                        tile_size, 
                                        tile_size
                                    )
                                    draw_rect_alpha(self.surface, (255, 215, 0, 50), rect)  # Semi-transparent gold

        # Draw energy nodes
        for i in range(params.max_energy_nodes):
            if state.energy_nodes_mask[i]:
                x, y = state.energy_nodes[i, :2]
                center_x = (x + 0.5) * tile_size
                center_y = (y + 0.5) * tile_size
                radius = (
                    tile_size // 4
                )  # Adjust this value to change the size of the circle
                pygame.draw.circle(
                    self.surface, (0, 255, 0, 255), (int(center_x), int(center_y)), radius
                )
        # Draw relic nodes
        for i in range(params.max_relic_nodes):
            if state.relic_nodes_mask[i]:
                x, y = state.relic_nodes[i, :2]
                rect_size = tile_size // 2  # Make the square smaller than the tile
                rect_x = x * tile_size + (tile_size - rect_size) // 2
                rect_y = y * tile_size + (tile_size - rect_size) // 2
                rect = pygame.Rect(rect_x, rect_y, rect_size, rect_size)
                pygame.draw.rect(self.surface, (173, 151, 32, 255), rect)  # Light blue color

       


        # Draw units
        for team in range(2):
            for i in range(params.max_units):
                if state.units_mask[team, i]:
                    x, y = state.units[team, i, :2]
                    center_x = (x + 0.5) * tile_size
                    center_y = (y + 0.5) * tile_size
                    radius = (
                        tile_size // 3
                    )  # Adjust this value to change the size of the circle
                    color = (
                        (255, 0, 0, 255) if team == 0 else (0, 0, 255, 255)
                    )  # Red for team 0, Blue for team 1
                    pygame.draw.circle(
                        self.surface, color, (int(center_x), int(center_y)), radius
                    )
        # Draw unit counts
        unit_counts = {}
        for team in range(2):
            for i in range(params.max_units):
                if state.units_mask[team, i]:
                    x, y = np.array(state.units[team, i, :2])
                    pos = (x, y)
                    if pos not in unit_counts:
                        unit_counts[pos] = 0
                    unit_counts[pos] += 1

        font = pygame.font.Font(None, 32)  # You may need to adjust the font size
        for pos, count in unit_counts.items():
            if count >= 1:
                x, y = pos
                text = font.render(str(count), True, (255, 255, 255))  # White text
                text_rect = text.get_rect(
                    center=((x + 0.5) * tile_size, (y + 0.5) * tile_size)
                )
                self.surface.blit(text, text_rect)

        # Draw the grid lines
        for x in range(params.map_width + 1):
            pygame.draw.line(
                self.surface,
                (100, 100, 100),
                (x * tile_size, 0),
                (x * tile_size, params.map_height * tile_size),
            )
        for y in range(params.map_height + 1):
            pygame.draw.line(
                self.surface,
                (100, 100, 100),
                (0, y * tile_size),
                (params.map_width * tile_size, y * tile_size),
            )

        
        self.screen.blit(self.surface, (0, 0))
        # Update the display
        pygame.display.flip()

        # # Handle events to keep the window responsive
        # for event in pygame.event.get():
        #     if event.type == pygame.TEXTINPUT and event.text == " ":
        #         while True:
        #             for event in pygame.event.get():
        #                 if event.type == pygame.TEXTINPUT and event.text == " ":
        #                     break

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
