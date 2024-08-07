from flax import struct
import chex
import jax

from luxai_s3.params import EnvParams
import jax.numpy as jnp

@struct.dataclass
class EnvState:
    units: chex.Array
    """Units in the environment with shape (2, N, 3) for 2 teams, N max units, and 3 features.

    3 features are for position (x, y), TODO (stao):
    """
    units_mask: chex.Array
    """Mask of units in the environment with shape (2, N) for 2 teams, N max units"""
    energy_nodes: chex.Array
    """Energy nodes in the environment with shape (N, 2) for N max energy nodes, and 2 features.

    2 features are for position (x, y)
    """
    energy_nodes_mask: chex.Array
    """Mask of energy nodes in the environment with shape (N) for N max energy nodes"""
    relic_nodes: chex.Array
    """Relic nodes in the environment with shape (N, 2) for N max relic nodes, and 3 features.

    3 features are for position (x, y), and a relic ID number. The relic ID number corresponds with a random relic configuration.
    """
    relic_nodes_mask: chex.Array
    """Mask of relic nodes in the environment with shape (2, N) for 2 teams, N max relic nodes"""

    map_features: chex.Array
    """Map features in the environment with shape (H, W, 2) for H height, W width, TODO (stao):
    """

    team_points: chex.Array
    """Team points in the environment with shape (2) for 2 teams"""


    steps: int = 0

@struct.dataclass
class EnvObs:
    """Observation of the environment. A subset of the environment state due to partial observability."""
    units: chex.Array
    units_mask: chex.Array
    """Mask of units in the environment with shape (2, N) for 2 teams, N max units"""


def state_to_flat_obs(state: EnvState) -> chex.Array:
    pass

def flat_obs_to_state(flat_obs: chex.Array) -> EnvState:
    pass


def gen_state(params: EnvParams) -> EnvState:
    generated = gen_map(params)
    state = EnvState(
        units = jnp.zeros(shape=(2, params.max_units, 3), dtype=jnp.int16),
        units_mask = jnp.zeros(shape=(2, params.max_units), dtype=jnp.int16),
        team_points = jnp.zeros(shape=(2), dtype=jnp.int16),
        energy_nodes = generated["energy_nodes"],
        energy_nodes_mask = generated["energy_nodes_mask"],
        relic_nodes = generated["relic_nodes"],
        relic_nodes_mask = generated["relic_nodes_mask"],
        map_features = generated["map_features"],
    )
    
    state = spawn_unit(state, 0, 0, [0, 0])
    state = spawn_unit(state, 0, 1, [0, 0])
    state = spawn_unit(state, 0, 2, [0, 0])
    state = spawn_unit(state, 1, 0, [15, 15])
    state = spawn_unit(state, 1, 1, [15, 15])
    state = spawn_unit(state, 1, 2, [15, 15])
    return state
def spawn_unit(state: EnvState, team: int, unit_id: int, position: chex.Array) -> EnvState:
    state = state.replace(units=state.units.at[team, unit_id, :].set([position[0], position[1], 0]))
    state = state.replace(units_mask=state.units_mask.at[team, unit_id].set(1))
    return state

def gen_map(params: EnvParams) -> chex.Array:
    map_features = jnp.zeros(shape=(params.map_height, params.map_width, 2), dtype=jnp.int16)
    energy_nodes = jnp.zeros(shape=(params.max_energy_nodes, 2), dtype=jnp.int16)
    energy_nodes_mask = jnp.zeros(shape=(params.max_energy_nodes), dtype=jnp.int16)
    relic_nodes = jnp.zeros(shape=(params.max_relic_nodes, 3), dtype=jnp.int16)
    relic_nodes_mask = jnp.zeros(shape=(params.max_relic_nodes), dtype=jnp.int16)
    if params.map_type == "dev0":
        assert params.map_height == 16 and params.map_width == 16
        map_features = map_features.at[4, 4, 0].set(1)
        map_features = map_features.at[:3, :2, 0].set(1)
        map_features = map_features.at[4:7, 6:9, 0].set(1)
        map_features = map_features.at[4, 5, 0].set(1)
        map_features = map_features.at[9:12, 5:6, 0].set(1)
        map_features = map_features.at[14:, 12:15, 0].set(1)
        
        map_features = map_features.at[11, 11, 0].set(1)
        map_features = map_features.at[11, 12, 0].set(1)
        energy_nodes = energy_nodes.at[0, :].set(jnp.array([4, 4], dtype=jnp.int16))
        energy_nodes_mask = energy_nodes_mask.at[0].set(1)
        energy_nodes = energy_nodes.at[1, :].set(jnp.array([11, 11], dtype=jnp.int16))
        energy_nodes_mask = energy_nodes_mask.at[1].set(1)


        relic_nodes = relic_nodes.at[0, :].set(jnp.array([1, 1, 0], dtype=jnp.int16))
        relic_nodes_mask = relic_nodes_mask.at[0].set(1)
        relic_nodes = relic_nodes.at[1, :].set(jnp.array([1, 4, 1], dtype=jnp.int16))
        relic_nodes_mask = relic_nodes_mask.at[1].set(1)
        relic_nodes = relic_nodes.at[2, :].set(jnp.array([14, 11, 0], dtype=jnp.int16))
        relic_nodes_mask = relic_nodes_mask.at[2].set(1)
        relic_nodes = relic_nodes.at[3, :].set(jnp.array([14, 14, 1], dtype=jnp.int16))
        relic_nodes_mask = relic_nodes_mask.at[3].set(1)

        
    return dict(map_features=map_features, energy_nodes=energy_nodes, relic_nodes=relic_nodes, energy_nodes_mask=energy_nodes_mask, relic_nodes_mask=relic_nodes_mask)
