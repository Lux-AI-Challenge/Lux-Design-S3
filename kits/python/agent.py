# from lux.kit import obs_to_game_state, GameState
# from lux.config import EnvConfig
from lux.utils import direction_to
import sys
import numpy as np
class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        # game_state = obs_to_game_state(step, self.env_cfg, obs)
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # basic strategy here is simply to have some units randomly explore and some units collecting as much energy as possible
        # and once a relic node is found, we send all units to move randomly around the first relic node to gain points
        # and information about where relic nodes are found are saved for the next match
        
        
        # save any new relic nodes that we discover for the rest of the game.
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])
            
        
        self.unit_explore_locations = dict()
        
        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            if len(self.relic_node_positions) > 0:
                nearest_relic_node_position = self.relic_node_positions[0]
                manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(unit_pos[1] - nearest_relic_node_position[1])
                if manhattan_distance <= 4:
                    random_direction = np.random.randint(0, 4)
                    actions[unit_id] = [random_direction, 0, 0]
                else:
                    actions[unit_id] = [direction_to(unit_pos, nearest_relic_node_position), 0, 0]
            else:
                if unit_id % 2 == 0:
                    # randomly explore by picking a random location on the map and moving there for about 20 steps
                    print(f"step: {step}", file=sys.stderr)
                    if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                        rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                        self.unit_explore_locations[unit_id] = rand_loc
                    actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
                else:
                    # follow energy field to its peak
                    for delta in np.array([[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]]):
                        next_pos = unit_pos + delta # (x, y) format
                        if next_pos[0] < 0 or next_pos[0] >= self.env_cfg["map_width"] or next_pos[1] < 0 or next_pos[1] >= self.env_cfg["map_height"]:
                            continue
                        next_pos_energy = obs["map_features"]["energy"][next_pos[0], next_pos[1]]
                        cur_pos_energy = obs["map_features"]["energy"][unit_pos[0], unit_pos[1]]
                        if next_pos_energy > cur_pos_energy:
                            actions[unit_id] = [direction_to(unit_pos, next_pos), 0, 0]
                            print(f"unit {unit_id} at {unit_pos} moving to {next_pos}, {next_pos_energy}, {cur_pos_energy}", file=sys.stderr)
                            break
        return actions
