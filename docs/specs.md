# Lux AI Season 3 Specs

## Background
TODO

## Environment

In the Lux AI Challenge Season 3, two teams compete against each other on a 2D map in a best of 5 match sequence (called a game). Both teams have a pool of units they can control to gain points around the map while also trying to prevent the other team from doing the same.

Unique to Season 3 is how various game mechanics and parameters are randomized at the start of each game and remain the same between matches in one game. Some mechanics/paramters include the map terrain/generation, how much units can see on the map, how might they be blocked by map features, etc. Each match is played with fog of war, where each team can only see what their own units can see, with everything else being hidden. Given that some mechanics are randomized between games, the specs will clearly document how they are randomized and what the possible values are. There is also a summary table of every game parameter that is randomized between games in the [Game Parameters](#game-parameters) section.

A core objective of this game is a balanced strategy of exploration and exploitation. It is recommended to explore more in the first match or two before leveraging gained knowledge to win the latter matches.

## Map

The map is a randomly generated 2D grid of size 16x16. There are several core features that make up the map: Empty Tiles, Asteroid Tiles, Nebula Tiles, Energy Nodes, and Relic Nodes. Notably, in a game, the map is never regenerated completely between matches. Whatever is the state of the map at the end of one match is what is used for the next match.

### Empty Tiles

These are empty tiles in space without anything special about them. Units and nodes can be placed/move onto these tiles.

### Asteroid Tiles
Asteroid tiles are impassable tiles that block anything from moving/spawning onto them.

### Nebula Tiles
Nebula tiles are passable tiles with a number of features

*Vision Reduction*: Nebula tiles can reduce/block vision of units. Because of vision reduction it is even possible for a unit to be unable to see itself while still being able to move! See [Vision](#vision) for more details on how team vision is determined. All nebula tiles have the same vision reduction value called `params.nebula_tile_vision_reduction` which is randomized from 0 to 3. 

### Energy Nodes

### Relic Nodes

## Units

Units in the game are ships that can move one tile in 4 directions (up, right, down, left) and perform a long energy sapping action. Units can overlap with other friendly units if they move onto the same tile.

Units have a single property called energy which determines whether they can perform actions and start with 100 energy and can have a max of 400 energy. Actions at game start have fixed energy costs.

The sap action lets a unit target a specific tile on the map within a range called `params.unit_sap_range` and reduces the energy of each unit on the target tile by `params.unit_sap_amount`. `params.unit_sap_amount` is random between 10 and 50, and `params.unit_sap_range` is random between 3 and 8.

Move action cost is `params.unit_move_cost` which is random between 1 and 5. Sap action cost is the same as the amount sapped which is `params.unit_sap_amount`, randomized between 10 and 50.


### Vision

A team's vision is the combined vision of all units on that team. Team vision is essentially a boolean mask / matrix over the 2D map indicating whether that tile's information is visible to the team (in code it is not just 0s / gibberish).

To determine which map tiles are visible to a unit, we compute a value known as vision power around the unit. The vision power of a tile is equal `1 + unit_vision_power - min(dx, dy)` where `dx` and `dy` are the absolute difference in the x and y coordinates between the unit and the tile. By default the unit vision power is equal to `params.unit_sensor_range`. The unit_vision_power value is reduced by `params.nebula_tile_vision_reduction` if the unit is on a nebula tile.

Nebula tiles have a vision reduction value of `params.nebula_tile_vision_reduction`. This number is reduced from every tile's vision power if that tile is a nebula tile.

For example, naturally without any nebula tiles the vision power values look like below and create a square of visibility around the unit.

TODO: insert small diagram

When a unit is near a nebula tile, it can't see as far due to nebula tiles reducing the vision power.

When a unit is inside a nebula tile, if the nebula vision reduction is powerful enough, the unit cannot see far if not anywhere at all.

## Relic Nodes and Team Points


## Match Resolution Order

At each time step of a match, we run the following steps in order:
1. Move all units that have enough energy to move
2. Execute the sap actions of all units that have enough energy to do so
3. Update the energy of all units based on their position
4. Compute new team points
5. Determine the team vision for all teams and return observations accordingly
6. Spawn units for all teams
7. Environment objects like asteroids/nebula tiles/energy nodes move around in space

## Game Parameters

### Randomized Game Parameters / Map Generation

There are a number of randomized game paramteres which can modify and even disable/enable certain game mechanics. None of these game parameters are changed between matches in a game. These parameters are also not given to the teams themselves and must be discovered through exploration.

- `params.unit_sap_amount` - 10 to 50
- `params.unit_sap_range` - 3 to 8
- `params.nebula_tile_vision_reduction` - 0 to 3
- `params.unit_sensor_range` - 1 to 3
- `params.unit_move_cost` - 1 to 5

## Using the Visualizer

The visualizer will display the state of the environment at time step `t` out of some max number indicated in the page under the map. Actions taken at timestep `t` will affect the state of the game and be reflected in the next timestep `t+1`.