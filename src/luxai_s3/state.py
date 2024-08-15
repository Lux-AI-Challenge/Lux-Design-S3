import chex
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
class EnvState:
    units: chex.Array
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

    energy_field: chex.Array
    """Energy field in the environment with shape (H, W) for H height, W width. This is generated from other state"""
    
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

    map_features: chex.Array
    """Map features in the environment with shape (H, W, 2) for H height, W width, TODO (stao):
    """

    sensor_mask: chex.Array
    """Sensor mask in the environment with shape (T, H, W) for T teams, H height, W width. This is generated from other state"""

    team_points: chex.Array
    """Team points in the environment with shape (T) for T teams"""

    steps: int = 0


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
        shape=(params.map_height, params.map_width), dtype=jnp.int16
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
                    relic_nodes_map_weights.at[y, x].add(relic_node_config[dy, dx]),
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
        units=jnp.zeros(shape=(params.num_teams, params.max_units, 3), dtype=jnp.int16),
        units_mask=jnp.zeros(
            shape=(params.num_teams, params.max_units), dtype=jnp.bool
        ),
        team_points=jnp.zeros(shape=(params.num_teams), dtype=jnp.int32),
        energy_nodes=generated["energy_nodes"],
        energy_node_fns=generated["energy_node_fns"],
        energy_nodes_mask=generated["energy_nodes_mask"],
        energy_field=jnp.zeros(shape=(params.map_height, params.map_width), dtype=jnp.int16),
        relic_nodes=generated["relic_nodes"],
        relic_nodes_mask=generated["relic_nodes_mask"],
        relic_node_configs=generated["relic_node_configs"],
        relic_nodes_map_weights=relic_nodes_map_weights,
        sensor_mask=jnp.zeros(
            shape=(params.num_teams, params.map_height, params.map_width),
            dtype=jnp.bool,
        ),
        map_features=generated["map_features"],
    )

    state = spawn_unit(state, 0, 0, [0, 0])
    state = spawn_unit(state, 0, 1, [0, 0])
    # state = spawn_unit(state, 0, 2, [0, 0])
    state = spawn_unit(state, 1, 0, [15, 15])
    state = spawn_unit(state, 1, 1, [15, 15])
    # state = spawn_unit(state, 1, 2, [15, 15])
    return state


def spawn_unit(
    state: EnvState, team: int, unit_id: int, position: chex.Array
) -> EnvState:
    state = state.replace(
        units=state.units.at[team, unit_id, :].set(
            jnp.array([position[0], position[1], 0], dtype=jnp.int16)
        )
    )
    state = state.replace(units_mask=state.units_mask.at[team, unit_id].set(True))
    return state


def gen_map(key: chex.PRNGKey, params: EnvParams) -> chex.Array:
    map_features = jnp.zeros(
        shape=(params.map_height, params.map_width, 2), dtype=jnp.int16
    )
    energy_nodes = jnp.zeros(shape=(params.max_energy_nodes, 2), dtype=jnp.int16)
    energy_nodes_mask = jnp.zeros(shape=(params.max_energy_nodes), dtype=jnp.int16)
    relic_nodes = jnp.zeros(shape=(params.max_relic_nodes, 2), dtype=jnp.int16)
    relic_nodes_mask = jnp.zeros(shape=(params.max_relic_nodes), dtype=jnp.int16)
    if params.map_type == "dev0":
        assert params.map_height == 16 and params.map_width == 16
        map_features = map_features.at[4, 4, 0].set(NEBULA_TILE)
        map_features = map_features.at[:3, :2, 0].set(NEBULA_TILE)
        map_features = map_features.at[4:7, 6:9, 0].set(NEBULA_TILE)
        map_features = map_features.at[4, 5, 0].set(NEBULA_TILE)
        map_features = map_features.at[9:12, 5:6, 0].set(NEBULA_TILE)
        map_features = map_features.at[14:, 12:15, 0].set(NEBULA_TILE)

        map_features = map_features.at[12:15, 8:10, 0].set(ASTEROID_TILE)
        map_features = map_features.at[1:4, 6:8, 0].set(ASTEROID_TILE)

        map_features = map_features.at[11:12, 3:6].set(ASTEROID_TILE)
        map_features = map_features.at[4:5, 10:13, 0].set(ASTEROID_TILE)

        map_features = map_features.at[11, 11, 0].set(NEBULA_TILE)
        map_features = map_features.at[11, 12, 0].set(NEBULA_TILE)
        energy_nodes = energy_nodes.at[0, :].set(jnp.array([4, 4], dtype=jnp.int16))
        energy_nodes_mask = energy_nodes_mask.at[0].set(1)
        energy_nodes = energy_nodes.at[1, :].set(jnp.array([11, 11], dtype=jnp.int16))
        energy_nodes_mask = energy_nodes_mask.at[1].set(1)
        energy_node_fns = jnp.array(
            [
                [0, 1, 0, 4],
                # [1, 4, 0, 2],
                [0, 1, 0, 4],
                # [1, 4, 0, 0]
            ]
        )
        energy_node_fns = jnp.concat([energy_node_fns, jnp.zeros((params.max_energy_nodes - 2, 4), dtype=jnp.int16)], axis=0)

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
