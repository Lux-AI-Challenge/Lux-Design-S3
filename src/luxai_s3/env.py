from typing import Any, Dict, Tuple, Union
import gymnax
from gymnax.environments import environment
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
    
    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Return observation from raw state trafo."""
        obs_end = jnp.zeros(shape=(self.size, self.size), dtype=jnp.float32)
        end_cond = state.row >= self.size
        obs_upd = obs_end.at[state.row, state.column].set(1.0)
        return jax.lax.select(end_cond, obs_end, obs_upd)

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
        raise NotImplementedError

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        raise NotImplementedError

    def state_space(self, params: EnvParams):
        """State space of the environment."""
        raise NotImplementedError