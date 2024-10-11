# Lux AI Season 3 Specs

For documentation on the API, see [this document](https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/kits/). To get started developing a bot, see [our Github](https://github.com/Lux-AI-Challenge/Lux-Design-S3/).

We are always looking for feedback and bug reports, if you find any issues with the code, specifications etc. please ping us on [Discord](https://discord.gg/aWJt3UAcgn) or post a [GitHub Issue](https://github.com/Lux-AI-Challenge/Lux-Design-S3/issues)

## Background
A story is being woven still...

## Environment

In the Lux AI Challenge Season 3, two teams compete against each other on a 2D map in a best of 5 match sequence (called a game) with each match lasting 100 time steps. Both teams have a pool of units they can control to gain points around the map while also trying to prevent the other team from doing the same.

Unique to Season 3 is how various game mechanics and parameters are randomized at the start of each game and remain the same between matches in one game. Some mechanics/paramters include the map terrain/generation, how much units can see on the map, how might they be blocked by map features, etc. Each match is played with fog of war, where each team can only see what their own units can see, with everything else being hidden. Given that some mechanics are randomized between games, the specs will clearly document how they are randomized and what the possible values are. There is also a summary table of every game parameter that is randomized between games in the [Game Parameters](#game-parameters) section.

A core objective of this game is a balanced strategy of exploration and exploitation. It is recommended to explore more in the first match or two before leveraging gained knowledge about the map and opponent behavior to win the latter matches.

## Map

The map is a randomly generated 2D grid of size 24x24. There are several core features that make up the map: Empty Tiles, Asteroid Tiles, Nebula Tiles, Energy Nodes, and Relic Nodes. Notably, in a game, the map is never regenerated completely between matches. Whatever is the state of the map at the end of one match is what is used for the next match.

### Empty Tiles

These are empty tiles in space without anything special about them. Units and nodes can be placed/move onto these tiles.

### Asteroid Tiles
Asteroid tiles are impassable tiles that block anything from moving/spawning onto them. These tiles might move around over time during the map in a symmetric fashion.

### Nebula Tiles
Nebula tiles are passable tiles with a number of features. These tiles might move around over time during the map in a symmetric fashion.

*Vision Reduction*: Nebula tiles can reduce/block vision of units. Because of vision reduction it is even possible for a unit to be unable to see itself while still being able to move! See [Vision](#vision) for more details on how team vision is determined. All nebula tiles have the same vision reduction value called `params.nebula_tile_vision_reduction` which is randomized from 0 to 3. 

### Energy Nodes

Energy nodes are mysterious objects that emit energy fields which can be harvested by units. These nodes might move around over time during the map in a symmetric fashion. In code, what actually occurs in each game is energy nodes are randomly generated on the map symmetrically and a random function is generated for each node. Each energy node's function is a function of distance. The energy value of a tile on a map is determined to be the sum of the energy node functions applied to the distance between tile and each node.
<!-- TODO link to code -->

### Relic Nodes

Relic nodes are objects in space that enable ships to go near it to gain team points. These relic nodes however are ancient and thus fragmented. As a result, only certain tiles near the relic nodes when a friendly ship is on it will gain points. The tiles that yield points are always hidden and can only be discovered by trial and error by moving around the relic nodes. Relic nodes themselves can be observed.

In code, a random 5x5 configuration / mask centered on the relic node is generated indicating which tiles yield points and which don't. Multiple ships can stack on one tile and all will gain a point each for their team per time step they remain on the tile. Note that ship stacking can be risky due to the [sapping action](#sap-actions).

## Units

Units in the game are ships that can move one tile in 5 directions (center, up, right, down, left) and perform a ranged energy sapping action. Units can overlap with other friendly units if they move onto the same tile. Units have a energy property which determines whether they can perform actions and start with 100 energy and can have a max of 400 energy. Energy is recharged via the energy field of the map.

### Move Actions

All move actions except moving center cost `params.unit_move_cost` energy to perform. Moving center is always free (a zero action). Attempting to move off the edge of the map results in no movement occuring but energy is still consumed. Units cannot move onto tiles with an impassible feature like an asteroid tile.

### Sap Actions

The sap action lets a unit target a specific tile on the map within a range called `params.unit_sap_range` and reduces the energy of each opposition unit on the target tile by `params.unit_sap_cost` while also costing `unit_sap_cost` energy to use. Moreover, any opposition units on the 8 adjacent tiles to the target tile are also sapped and their energy is reduced by `params.unit_sap_cost * params.unit_sap_dropoff_factor`.

Sap actions are submitted to the game engine / environment as a delta x and delta y value relative to the unit's current position. The delta x and delta y value magnitudes must both be <= `params.unit_sap_range`, so the sap range is a square around the unit.

Generally sap actions are risky since a single miss means your ships lose energy while the opponent does not. The area of effect can mitigate this risk somewhat depending on game parameters. Sap actions can however prove very valuable when opposition ships are heavily stacked and get hit as sapping the stacked tile hits every ship on the tile.


<!-- Move action cost is `params.unit_move_cost` which is random between 1 and 5. Sap action cost is the same as the amount sapped which is `params.unit_sap_amount`, randomized between 10 and 50. -->


### Vision

A team's vision is the combined vision of all units on that team. Team vision is essentially a boolean mask / matrix over the 2D map indicating whether that tile's information is visible to the team (in code it is not just 0s / gibberish). In this game, you can think of each unit having an "eye in the sky" sattelite that is capturing information about the units surroundings, but this sattelite has reduced accuracy the farther away the tile is from the unit.

To determine which map tiles are visible to a unit, we compute a value known as vision power around the unit. The vision power of a tile is equal `1 + unit_vision_power - min(dx, dy)` where `dx` and `dy` are the absolute difference in the x and y coordinates between the unit and the tile. By default the unit vision power is equal to `params.unit_sensor_range`. The `unit_vision_power` value is reduced by `params.nebula_tile_vision_reduction` if the unit is on a nebula tile.

Nebula tiles have a vision reduction value of `params.nebula_tile_vision_reduction`. This number is reduced from every tile's vision power if that tile is a nebula tile.

For example, naturally without any nebula tiles the vision power values look like below and create a square of visibility around the unit.

TODO: insert small diagram

When a unit is near a nebula tile, it can't see details about some nebula tiles, but it can see tiles beyond nebula tiles.

When a unit is inside a nebula tile, if the nebula vision reduction is powerful enough, the unit cannot see far if not anywhere at all.

## Win Conditions

To win the game, the team must have won the most matches out of the 5 match sequence.

To win a match, the team must have gained more relic points than the other team at the end of the match. If the relic points scores are tied, then the match winner is decided by who has more total unit energy. If that is also tied then the winner is chosen at random.


## Match Resolution Order

At each time step of a match, we run the following steps in order:
1. Move all units that have enough energy to move
2. Execute the sap actions of all units that have enough energy to do so
3. Update the energy of all units based on their position
4. Compute new team points
5. Determine the team vision for all teams and return observations accordingly
6. Spawn units for all teams. Remove units that have less than 0 energy due to saps.
7. Environment objects like asteroids/nebula tiles/energy nodes move around in space

## Game Parameters

The full set of game parameters can be found here in the codebase: https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/src/luxai_s3/params.py

### Randomized Game Parameters / Map Generation

There are a number of randomized game paramteres which can modify and even disable/enable certain game mechanics. None of these game parameters are changed between matches in a game. These parameters are also not given to the teams themselves and must be discovered through exploration.

- `params.unit_sap_amount` - 10 to 50
- `params.unit_sap_range` - 3 to 8
- `params.nebula_tile_vision_reduction` - 0 to 3
- `params.unit_sensor_range` - 1 to 3
- `params.unit_move_cost` - 1 to 5

These parameter ranges (and other parameters) are subject to change in the beta phase of this competition as we gather feedback and data.

## Using the Visualizer

The visualizer will display the state of the environment at time step `t` out of some max number indicated in the page under the map. Actions taken at timestep `t` will affect the state of the game and be reflected in the next timestep `t+1`.