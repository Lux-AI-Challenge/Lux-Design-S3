import math
import sys
from typing import List
import numpy as np
from dataclasses import dataclass
from lux.cargo import UnitCargo
from lux.config import EnvConfig

# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])

@dataclass
class Unit:
    team_id: int
    unit_id: str
    pos: np.ndarray
    power: int
    cargo: UnitCargo
    env_cfg: EnvConfig
    unit_cfg: dict
    action_queue: List

    @property
    def agent_id(self):
        if self.team_id == 0: return "player_0"
        return "player_1"

    def move(self, direction, repeat=0, n=1):
        if isinstance(direction, int):
            direction = direction
        else:
            pass
        return np.array([0, direction, 0, 0, repeat, n])

    def __str__(self) -> str:
        out = f"[{self.team_id}] {self.unit_id} {self.unit_type} at {self.pos}"
        return out