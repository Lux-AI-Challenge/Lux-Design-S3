from flax import struct


@struct.dataclass
class EnvParams:
    max_steps_in_match: int = 100
    map_type: str = "random"
    """Map generation algorithm. Can change between games"""
    map_width: int = 16
    map_height: int = 16
    num_teams: int = 2
    match_count_per_episode: int = 5
    """number of matches to play in one episode"""

    # configs for units
    max_units: int = 10
    init_unit_energy: int = 100
    min_unit_energy: int = 0
    max_unit_energy: int = 400
    unit_move_cost: int = 2


    unit_sap_cost: int = 10
    """
    The unit sap cost is the amount of energy a unit uses when it saps another unit. Can change between games.
    """
    unit_sap_drain: int = 1
    """
    The unit sap drain is the amount of energy a unit drains from another unit when it saps it. Can change between games.
    """
    unit_sap_range: int = 5
    """
    The unit sap range is the range of the unit's sap action.
    """


    # configs for energy nodes
    max_energy_nodes: int = 10
    max_energy_per_tile: int = 20
    min_energy_per_tile: int = -20


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
