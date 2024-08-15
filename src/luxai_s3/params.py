from flax import struct


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 200
    map_type: str = "random"
    map_width: int = 16
    map_height: int = 16
    num_teams: int = 2
    max_units: int = 4
    init_unit_energy: int = 10

    # configs for energy nodes
    max_energy_nodes: int = 10


    max_relic_nodes: int = 10
    relic_config_size: int = 5
    fog_of_war: bool = True
    """
    whether there is fog of war or not
    """
    unit_sensor_range: int = 2
    """
    The unit sensor range is the range of the unit's sensor.
    Units provide "vision power" over tiles in range, equal to manhattan distance to the unit.

    vision power > 0 that team can see the tiles properties
    """

    # nebula tile params
    nebula_tile_vision_reduction: int = 1
    """
    The nebula tile vision reduction is the amount of vision reduction a nebula tile provides.
    A tile can be seen if the vision power over it is > 0.
    """
