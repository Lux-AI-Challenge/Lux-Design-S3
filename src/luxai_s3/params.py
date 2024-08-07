from flax import struct


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 200
    map_type: str = "random"
    map_width: int = 16
    map_height: int = 16
    max_units: int = 4
    max_energy_nodes: int = 100
    max_relic_nodes: int = 100
    fog_of_war: bool = False # TODO (stao): support this
    unit_sensor_range: int = 4 # TODO (stao): support this