const readline = require('readline');
const { directionTo } = require('./lux/utils');

class Agent {
    constructor(player, env_cfg) {
        this.player = player;
        this.opp_player = this.player === "player_0" ? "player_1" : "player_0";
        this.team_id = this.player === "player_0" ? 0 : 1;
        this.opp_team_id = this.team_id === 0 ? 1 : 0;
        this.env_cfg = env_cfg;
        
        this.relic_node_positions = [];
        this.discovered_relic_nodes_ids = new Set();
        this.unit_explore_locations = {};
    }

    act(step, obs, remainingOverageTime = 60) {
        const unit_mask = obs.units_mask[this.team_id];
        const unit_positions = obs.units.position[this.team_id];
        const unit_energys = obs.units.energy[this.team_id];
        const observed_relic_node_positions = obs.relic_nodes;
        const observed_relic_nodes_mask = obs.relic_nodes_mask;
        const team_points = obs.team_points;
        
        const available_unit_ids = unit_mask.reduce((acc, val, idx) => val ? [...acc, idx] : acc, []);
        const visible_relic_node_ids = observed_relic_nodes_mask.reduce((acc, val, idx) => val ? acc.add(idx) : acc, new Set());
        
        const actions = Array(this.env_cfg.max_units).fill().map(() => [0, 0, 0]);

        for (const id of visible_relic_node_ids) {
            if (!this.discovered_relic_nodes_ids.has(id)) {
                this.discovered_relic_nodes_ids.add(id);
                this.relic_node_positions.push(observed_relic_node_positions[id]);
            }
        }

        for (const unit_id of available_unit_ids) {
            const unit_pos = unit_positions[unit_id];
            const unit_energy = unit_energys[unit_id];
            if (this.relic_node_positions.length > 0) {
                const nearest_relic_node_position = this.relic_node_positions[0];
                const manhattan_distance = Math.abs(unit_pos[0] - nearest_relic_node_position[0]) + Math.abs(unit_pos[1] - nearest_relic_node_position[1]);
                
                if (manhattan_distance <= 4) {
                    const random_direction = Math.floor(Math.random() * 5);
                    actions[unit_id] = [random_direction, 0, 0];
                } else {
                    actions[unit_id] = [directionTo(unit_pos, nearest_relic_node_position), 0, 0];
                }
            } else {
                if (step % 20 === 0 || !(unit_id in this.unit_explore_locations)) {
                    const rand_loc = [Math.floor(Math.random() * this.env_cfg.map_width), Math.floor(Math.random() * this.env_cfg.map_height)];
                    this.unit_explore_locations[unit_id] = rand_loc;
                }
                actions[unit_id] = [directionTo(unit_pos, this.unit_explore_locations[unit_id]), 0, 0];
            }
        }
        return actions;
    }
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false
});

let agent;

rl.on('line', (line) => {
  const input = JSON.parse(line);
  if (!agent) {
    agent = new Agent(input.player, input.info.env_cfg);
  }
  const actions = agent.act(input.step, input.obs, input.remainingOverageTime);
  console.log(JSON.stringify({action: actions}));
});
