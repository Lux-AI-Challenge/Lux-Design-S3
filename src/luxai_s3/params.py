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
    max_energy_nodes: int = 10
    max_relic_nodes: int = 10
    relic_config_size: int = 5
    fog_of_war: bool = False
    unit_sensor_range: int = 2
