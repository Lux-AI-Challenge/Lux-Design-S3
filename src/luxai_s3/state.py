import chex
import flax
import jax
import jax.numpy as jnp
from flax import struct

from luxai_s3.params import EnvParams
EMPTY_TILE = 0
NEBULA_TILE = 1
ASTEROID_TILE = 2

ENERGY_NODE_FNS = [
    lambda d, x, y, z: jnp.sin(d * x + y) * z, lambda d, x, y, z: (x / (d + 1) + y) * z
]

@struct.dataclass
class UnitState:
    position: chex.Array
    """Position of the unit with shape (2) for x, y"""
    energy: int
    """Energy of the unit"""

@struct.dataclass
class MapTile:
    energy: int
    """Energy of the tile, generated via energy_nodes and energy_node_fns"""
    tile_type: int
    """Type of the tile"""

@struct.dataclass
class EnvState:
    units: UnitState
    """Units in the environment with shape (T, N, 3) for T teams, N max units, and 3 features.

    3 features are for position (x, y), and energy
    """
    units_mask: chex.Array
    """Mask of units in the environment with shape (T, N) for T teams, N max units"""
    energy_nodes: chex.Array
    """Energy nodes in the environment with shape (N, 2) for N max energy nodes, and 2 features.

    2 features are for position (x, y)
    """
    
    energy_node_fns: chex.Array
    """Energy node functions for computing the energy field of the map. They describe the function with a sequence of numbers
    
    The first number is the function used. The subsequent numbers parameterize the function. The function is applied to distance of map tile to energy node and the function parameters.
    """

    # energy_field: chex.Array
    # """Energy field in the environment with shape (H, W) for H height, W width. This is generated from other state"""
    
    energy_nodes_mask: chex.Array
    """Mask of energy nodes in the environment with shape (N) for N max energy nodes"""
    relic_nodes: chex.Array
    """Relic nodes in the environment with shape (N, 2) for N max relic nodes, and 2 features.

    2 features are for position (x, y)
    """
    relic_node_configs: chex.Array
    """Relic node configs in the environment with shape (N, K, K) for N max relic nodes and a KxK relic configuration"""
    relic_nodes_mask: chex.Array
    """Mask of relic nodes in the environment with shape (T, N) for T teams, N max relic nodes"""
    relic_nodes_map_weights: chex.Array
    """Map of relic nodes in the environment with shape (H, W) for H height, W width. True if a relic node is present, False otherwise. This is generated from other state"""

    map_features: MapTile
    """Map features in the environment with shape (W, H, 2) for W width, H height
    """

    sensor_mask: chex.Array
    """Sensor mask in the environment with shape (T, H, W) for T teams, H height, W width. This is generated from other state"""

    vision_power_map: chex.Array
    """Vision power map in the environment with shape (T, H, W) for T teams, H height, W width. This is generated from other state"""

    team_points: chex.Array
    """Team points in the environment with shape (T) for T teams"""

    steps: int = 0
    """steps taken in the environment"""
    match_steps: int = 0
    """steps taken in the current match"""


def serialize_env_states(env_states: list[EnvState]):
    def serialize_array(arr, key_path: str = ""):
        if key_path in ["vision_power_map", "sensor_mask", "relic_nodes_mask", "relic_node_configs", "energy_node_fns"]:
            return None
        if isinstance(arr, jnp.ndarray):
            return arr.tolist()
        elif isinstance(arr, dict):
            ret = dict()
            for k, v in arr.items():
                new_key = key_path + "/" + k if key_path else k
                new_val = serialize_array(v, new_key)
                if new_val is not None:
                    ret[k] = new_val
            return ret

        return arr
    steps = []
    for state in env_states:
        state = flax.serialization.to_state_dict(state)
        steps.append(serialize_array(state))

    return steps

def serialize_env_actions(env_actions: list):
    def serialize_array(arr, key_path: str = ""):
        if key_path in ["vision_power_map", "sensor_mask", "relic_nodes_mask", "relic_node_configs", "energy_node_fns"]:
            return None
        if isinstance(arr, jnp.ndarray):
            return arr.tolist()
        elif isinstance(arr, dict):
            ret = dict()
            for k, v in arr.items():
                new_key = key_path + "/" + k if key_path else k
                new_val = serialize_array(v, new_key)
                if new_val is not None:
                    ret[k] = new_val
            return ret

        return arr
    steps = []
    for state in env_actions:
        state = flax.serialization.to_state_dict(state)
        steps.append(serialize_array(state))

    return steps


@struct.dataclass
class EnvObs:
    """Observation of the environment. A subset of the environment state due to partial observability."""

    units: chex.Array
    units_mask: chex.Array
    """Mask of units in the environment with shape (T, N) for T teams, N max units"""


def state_to_flat_obs(state: EnvState) -> chex.Array:
    pass


def flat_obs_to_state(flat_obs: chex.Array) -> EnvState:
    pass


def gen_state(key: chex.PRNGKey, params: EnvParams) -> EnvState:
    generated = gen_map(key, params)
    relic_nodes_map_weights = jnp.zeros(
        shape=(params.map_width, params.map_height), dtype=jnp.int16
    )

    # TODO (this could be optimized better)
    def update_relic_node(relic_nodes_map_weights, relic_data):
        relic_node, relic_node_config, mask = relic_data
        start_y = relic_node[1] - params.relic_config_size // 2
        start_x = relic_node[0] - params.relic_config_size // 2
        for dy in range(params.relic_config_size):
            for dx in range(params.relic_config_size):
                y, x = start_y + dy, start_x + dx
                valid_pos = jnp.logical_and(
                    jnp.logical_and(y >= 0, x >= 0),
                    jnp.logical_and(y < params.map_height, x < params.map_width),
                )
                relic_nodes_map_weights = jnp.where(
                    valid_pos & mask,
                    relic_nodes_map_weights.at[x, y].add(relic_node_config[dy, dx]),
                    relic_nodes_map_weights,
                )
        return relic_nodes_map_weights, None

    # this is really slow...
    relic_nodes_map_weights, _ = jax.lax.scan(
        update_relic_node,
        relic_nodes_map_weights,
        (
            generated["relic_nodes"],
            generated["relic_node_configs"],
            generated["relic_nodes_mask"],
        ),
    )
    state = EnvState(
        units=UnitState(position=jnp.zeros(shape=(params.num_teams, params.max_units, 2), dtype=jnp.int16), energy=jnp.zeros(shape=(params.num_teams, params.max_units, 1), dtype=jnp.int16)),
        units_mask=jnp.zeros(
            shape=(params.num_teams, params.max_units), dtype=jnp.bool
        ),
        team_points=jnp.zeros(shape=(params.num_teams), dtype=jnp.int32),
        energy_nodes=generated["energy_nodes"],
        energy_node_fns=generated["energy_node_fns"],
        energy_nodes_mask=generated["energy_nodes_mask"],
        # energy_field=jnp.zeros(shape=(params.map_height, params.map_width), dtype=jnp.int16),
        relic_nodes=generated["relic_nodes"],
        relic_nodes_mask=generated["relic_nodes_mask"],
        relic_node_configs=generated["relic_node_configs"],
        relic_nodes_map_weights=relic_nodes_map_weights,
        sensor_mask=jnp.zeros(
            shape=(params.num_teams, params.map_height, params.map_width),
            dtype=jnp.bool,
        ),
        vision_power_map=jnp.zeros(shape=(params.num_teams, params.map_height, params.map_width), dtype=jnp.int16),
        map_features=generated["map_features"],
    )

    state = spawn_unit(state, 0, 0, [0, 0], params)
    state = spawn_unit(state, 0, 1, [0, 0], params)
    # state = spawn_unit(state, 0, 2, [0, 0])
    state = spawn_unit(state, 1, 0, [15, 15], params)
    state = spawn_unit(state, 1, 1, [15, 15], params)
    # state = spawn_unit(state, 1, 2, [15, 15])
    return state


def spawn_unit(
    state: EnvState, team: int, unit_id: int, position: chex.Array, params: EnvParams
) -> EnvState:
    unit_state = state.units
    unit_state = unit_state.replace(position=unit_state.position.at[team, unit_id, :].set(jnp.array(position, dtype=jnp.int16)))
    unit_state = unit_state.replace(energy=unit_state.energy.at[team, unit_id, :].set(jnp.array([params.init_unit_energy], dtype=jnp.int16)))
    # state = state.replace(
    #     units
    #     # units=state.units.at[team, unit_id, :].set(
    #     #     jnp.array([position[0], position[1], 0], dtype=jnp.int16)
    #     # )
    # )
    state = state.replace(units=unit_state, units_mask=state.units_mask.at[team, unit_id].set(True))
    return state

def set_tile(map_features: MapTile, x: int, y: int, tile_type: int) -> MapTile:
    return map_features.replace(tile_type=map_features.tile_type.at[x, y].set(tile_type))


def gen_map(key: chex.PRNGKey, params: EnvParams) -> chex.Array:
    map_features = MapTile(energy=jnp.zeros(
        shape=(params.map_height, params.map_width), dtype=jnp.int16
    ), tile_type=jnp.zeros(
        shape=(params.map_height, params.map_width), dtype=jnp.int16
    ))
    energy_nodes = jnp.zeros(shape=(params.max_energy_nodes, 2), dtype=jnp.int16)
    energy_nodes_mask = jnp.zeros(shape=(params.max_energy_nodes), dtype=jnp.int16)
    relic_nodes = jnp.zeros(shape=(params.max_relic_nodes, 2), dtype=jnp.int16)
    relic_nodes_mask = jnp.zeros(shape=(params.max_relic_nodes), dtype=jnp.int16)
    if params.map_type == "dev0":
        assert params.map_height == 16 and params.map_width == 16
        map_features = set_tile(map_features, 4, 4, NEBULA_TILE)
        map_features = set_tile(map_features, slice(3, 6), slice(2, 4), NEBULA_TILE)
        map_features = set_tile(map_features, slice(4, 7), slice(6, 9), NEBULA_TILE)
        map_features = set_tile(map_features, 4, 5, NEBULA_TILE)
        map_features = set_tile(map_features, slice(9, 12), slice(5, 6), NEBULA_TILE)
        map_features = set_tile(map_features, slice(14, 16), slice(12, 15), NEBULA_TILE)

        map_features = set_tile(map_features, slice(12, 15), slice(8, 10), ASTEROID_TILE)
        map_features = set_tile(map_features, slice(1, 4), slice(6, 8), ASTEROID_TILE)

        map_features = set_tile(map_features, slice(11, 12), slice(3, 6), ASTEROID_TILE)
        map_features = set_tile(map_features, slice(4, 5), slice(10, 13), ASTEROID_TILE)
        map_features = set_tile(map_features,15, 0, ASTEROID_TILE)

        map_features = set_tile(map_features, 11, 11, NEBULA_TILE)
        map_features = set_tile(map_features, 11, 12, NEBULA_TILE)
        energy_nodes = energy_nodes.at[0, :].set(jnp.array([4, 4], dtype=jnp.int16))
        energy_nodes_mask = energy_nodes_mask.at[0].set(1)
        energy_nodes = energy_nodes.at[1, :].set(jnp.array([11, 11], dtype=jnp.int16))
        energy_nodes_mask = energy_nodes_mask.at[1].set(1)
        energy_node_fns = jnp.array(
            [
                [0, 1.2, 1, 4],
                # [1, 4, 0, 2],
                [0, 1.2, 1, 4],
                # [1, 4, 0, 0]
            ]
        )
        energy_node_fns = jnp.concat([energy_node_fns, jnp.zeros((params.max_energy_nodes - 2, 4), dtype=jnp.float32)], axis=0)

        relic_nodes = relic_nodes.at[0, :].set(jnp.array([1, 1], dtype=jnp.int16))
        relic_nodes_mask = relic_nodes_mask.at[0].set(1)
        relic_nodes = relic_nodes.at[1, :].set(jnp.array([2, 13], dtype=jnp.int16))
        relic_nodes_mask = relic_nodes_mask.at[1].set(1)
        relic_nodes = relic_nodes.at[2, :].set(jnp.array([14, 14], dtype=jnp.int16))
        relic_nodes_mask = relic_nodes_mask.at[2].set(1)
        relic_nodes = relic_nodes.at[3, :].set(jnp.array([13, 2], dtype=jnp.int16))
        relic_nodes_mask = relic_nodes_mask.at[3].set(1)

        relic_node_configs = (
            jax.random.randint(
                key,
                shape=(
                    params.max_relic_nodes,
                    params.relic_config_size,
                    params.relic_config_size,
                ),
                minval=0,
                maxval=10,
                dtype=jnp.int16,
            )
            >= 6
        )

    return dict(
        map_features=map_features,
        energy_nodes=energy_nodes,
        energy_node_fns=energy_node_fns,
        relic_nodes=relic_nodes,
        energy_nodes_mask=energy_nodes_mask,
        relic_nodes_mask=relic_nodes_mask,
        relic_node_configs=relic_node_configs,
    )
