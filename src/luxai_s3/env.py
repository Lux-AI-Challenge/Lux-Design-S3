import functools
from typing import Any, Dict, Optional, Tuple, Union
import gymnax
from gymnax.environments import environment, spaces
import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
@struct.dataclass
class EnvState:
    steps: int = 0

@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 200

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
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray,Dict[Any, Any]]:
        # state = state.replace() # TODO (stao)
        reward = jnp.array(0.0)
        terminated = self.is_terminal(state, params)
        truncated = state.steps >= params.max_steps_in_episode
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            terminated,
            truncated,
            {"discount": self.discount(state, params)},
        )
    
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        
        state = EnvState(
            # TODO (stao)
        )

        return self.get_obs(state), state
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, terminated, truncated, info = self.step_env(key, state, action, params)
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on done
        done = terminated | truncated
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        info["final_observation"] = obs_st
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, terminated, truncated, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        obs, state = self.reset_env(key, params)
        return obs, state
    
    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros(shape=(3, 3, 2), dtype=jnp.float32)
        return obs
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
        raise NotImplementedError

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        return spaces.Discrete(10)

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        return spaces.Discrete(10)

    def state_space(self, params: EnvParams):
        """State space of the environment."""
        return spaces.Discrete(10)