export enum RobotType {
  Light,
  Heavy,
}

export enum Resource {
  Ice,
  Ore,
  Water,
  Metal,
  Power,
}

export enum Direction {
  Center,
  Up,
  Right,
  Down,
  Left,
}

export enum Faction {
  None = 'None',
  AlphaStrike = 'AlphaStrike',
  MotherMars = 'MotherMars',
  TheBuilders = 'TheBuilders',
  FirstMars = 'FirstMars',
}

export interface Tile {
  x: number;
  y: number;
}
export enum TileType {
  Space,
  Nebula,
  Asteroid,
}

export interface Cargo {
  ice: number;
  ore: number;
  water: number;
  metal: number;
}

export interface Action {
  type: string;
}

export interface MoveAction extends Action {
  type: 'move';
  direction: Direction;
}

export interface SapAction extends Action {
  type: 'sap';
  target: Tile;
  validSap: boolean;
}
export type RobotAction = MoveAction | SapAction

export interface Board {
  energy: number[][];
  energyNodes: number[][];
  tileType: number[][];
  relicNodes: number[][];
  relicNodeConfigs: number[][][];
  visionPowerMap: number[][][];
}

export interface Unit {
  unitId: string;
  tile: Tile;
}

export interface Robot extends Unit {
  action: RobotAction | null;
  prevAction: RobotAction | null;
  energy: number;
}

export interface Team {
  name: string;
  points: number;
  wins: number;
  robots: Robot[];
  sensorMask: boolean[][];
  error: string | null;
}

export interface Step {
  step: number;
  board: Board;
  teams: [Team, Team];
}

export interface Episode {
  steps: Step[];
  metadata: EpisodeMetadata;
  params: EnvParams;
}
/* eslint-disable */

export interface EnvParams {
  max_steps_in_match: number;
  map_type: string;
  map_width: number;
  map_height: number;
  num_teams: number;
  match_count_per_episode: number;
  max_units: number;
  init_unit_energy: number;
  min_unit_energy: number;
  max_unit_energy: number;
  unit_move_cost: number;
  unit_sap_cost: number;
  unit_sap_range: number;
  max_energy_nodes: number;
  max_energy_per_tile: number;
  min_energy_per_tile: number;
  max_relic_nodes: number;
  relic_config_size: number;
  fog_of_war: boolean;
  unit_sensor_range: number;
  nebula_tile_vision_reduction: number;
}
/* eslint-enable */

export interface EpisodeMetadata {
  teamNames: [string, string];
  seed?: number;
}
