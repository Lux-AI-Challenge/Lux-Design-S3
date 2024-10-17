# Lux AI Challenge Season 3 Kit

This folder contains the kits for the Lux AI Challenge Season 3.

In each starter kit folder we give you all the tools necessary to compete. Make sure to read the README document carefully. For debugging, you may log to standard error e.g. `console.error("hello")` or `print("hello", file=sys.stderr)`, and will be recorded by the competition servers / printed out.

## Kit Structure

Each agent is a folder of files with a `main.py` file and a language dependent `agent.py/agent.js/agent...` file. You can generally ignore `main.py` and focus on the `agent` file which is where you write your logic. For the rest of this document we will use a python based syntax and assume you are working with the python kit but the instructions apply to all kits.

In the `agent.py` file, we define a simple class that holds your agent, you can modify as you wish. You just need an `act` function to be called for each step of the game to generate your actions for your team. The `act` function has parameters `step` equal to the environment time step, `obs` equal to the actual observations, and `remainingOverageTime` representing how much extra time you have left before having to make a decision.

Example code is provided for how to read the observation data and return actions to submit to the environment.

## Environment Actions

The action space of the game is always a fixed `(N, 3)` array of integers to control up to units `0` to `N-1` on your team where `N` is the max number of units each team can have (example code shows how to determine `N`). At any given point in time you might not have `N` units on your team so actions for those nonexistent units do not do anything.

For each unit's action, the first integer indicates the type of action, which can be 0 for doing nothing, 1 to move up, 2 to move right, 3 to move down, 4 to move left, and 5 to sap a tile. The next 2 integers are only for the sap action and indicate the location of the tile to sap from relative to the unit's position (a delta x and y value).

## Observations

Each game/episode consists of a sequence of matches. The very first observation of each match is either empty (due to no units being spawned in yet to observe map details) or contains the previous match's final observation. Taking actions on the very first observation of the game or the final observation of a match will not do anything.

The game engine sends a raw JSON in the form of a string to each agent. When the JSON is parsed it has the following structure:

```js
// T is the number of teams (default is 2)
// N is the max number of units per team
// W, H are the width and height of the map
// R is the max number of relic nodes
{
  "obs": {
    "units": {
      "position": Array(T, N, 2),
      "energy": Array(T, N, 1)
    },
    // whether the unit exists and is visible to you. units_mask[t][i] is whether team t's unit i can be seen and exists.
    "units_mask": Array(T, N),
    // whether the tile is visible to the unit for that team
    "sensor_mask": Array(W, H),
    "map_features": {
        // amount of energy on the tile
        "energy": Array(W, H),
        // type of the tile. 0 is empty, 1 is a nebula tile, 2 is asteroid
        "tile_type": Array(W, H)
    },
    // whether the relic node exists and is visible to you.
    "relic_nodes_mask": Array(R),
    // position of the relic nodes.
    "relic_nodes": Array(R, 2),
    // points scored by each team in the current match
    "team_points": Array(T),
    // number of wins each team has in the current game/episode
    "team_wins": Array(T),
    // number of steps taken in the current game/episode
    "steps": int,
    // number of steps taken in the current match
    "match_steps": int
  },
  // number of steps taken in the current game/episode
  "remainingOverageTime": int, // total amount of time your bot can use whenever it exceeds 2s in a turn
  "player": str, // your player id
  "info": {
    "env_cfg": dict // some of the game's visible parameters
  }
}
```
Numbers are filled with -1 if the information is not visible to you. This includes "relic_nodes", "map_features.energy/tile_type", and "units.energy/position". This is determined based on the "sensor_mask" array given to your agent. If the map feature, relic node position, or unit position is in the sensor mask, then the values are the real values, otherwise they are -1.

Moreover, all map sized arrays with shape (W, H) are accessed by the convention of `array[x][y]` where `x` is along the width, and `y` is along the height of the map.

Game parameters are given to agents but not all are visible to the agent. The only ones visible are map_width, map_height, max_steps_in_match, match_count_per_episode, unit_move_cost, unit_sap_cost, unit_sap_range. The rest will have to be inferred by your agent.

Example of what the observation JSON string looks like when parsed as JSON is in the sample_step_0_input.txt and sample_step_input.txt files in this folder.