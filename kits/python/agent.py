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
        self.unit_explore_locations = dict()
        self.exploring_units = []
        
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team
        
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)


        # basic strategy here is simply to have some units randomly explore and some units collecting as much energy as possible
        # and once a relic node is found, we send all units to move randomly around the first relic node to gain points
        # and information about where relic nodes are found are saved for the next match
        
        # save any new relic nodes that we discover for the rest of the game.

        total_units = len(available_unit_ids)
        if total_units >2:
            num_explorers = total_units // 3
        else:
            num_explorers = 0

        while len(self.exploring_units) < num_explorers:
            # Filter units that are not already explorers
            potential_explorers = [u for u in available_unit_ids if u not in self.exploring_units]
            
            # If no units are left to assign as explorers, break the loop
            if not potential_explorers:
                break

            # Randomly assign one of the remaining units as a new explorer
            new_explorer = np.random.choice(potential_explorers)
            self.exploring_units.append(new_explorer)

        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])
            

        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            if unit_energy == 0:
                actions[unit_id] = [0, 0, 0]

            elif unit_id in self.exploring_units:
                # Explore strategy
                if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]),
                                np.random.randint(0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
            
            elif len(self.relic_node_positions) > 0:
                nearest_relic_node_position = min(
                        self.relic_node_positions,
                        key=lambda relic_pos: abs(unit_pos[0] - relic_pos[0]) + abs(unit_pos[1] - relic_pos[1])
                    )
                manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(unit_pos[1] - nearest_relic_node_position[1])
                
                # if close to the relic node we want to hover around it and hope to gain points

                #Vishak
                if manhattan_distance <= 4:
                    valid_directions = [0]  # Start with the center as always valid
                    if unit_pos[1] > 0:  # Not in the top border
                        valid_directions.append(1)  # Add "up"
                    if unit_pos[0] < self.env_cfg["map_width"] - 1:  # Not in the right border
                        valid_directions.append(2)  # Add "right"
                    if unit_pos[1] < self.env_cfg["map_width"] - 1:  # Not in the bottom border
                        valid_directions.append(3)  # Add "down"
                    if unit_pos[0] > 0:  # Not in the left border
                        valid_directions.append(4)  # Add "left"
                    
                    random_direction = np.random.choice(valid_directions)
                    actions[unit_id] = [random_direction, 0, 0]

                else:
                    # otherwise we want to move towards the relic node
                    actions[unit_id] = [direction_to(unit_pos, nearest_relic_node_position), 0, 0]
            else:
                # randomly explore by picking a random location on the map and moving there for about 20 steps
                if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
        return actions
