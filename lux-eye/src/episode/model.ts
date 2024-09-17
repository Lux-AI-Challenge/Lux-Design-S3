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

export interface BidAction extends Action {
  type: 'bid';
  bid: number;
  faction: Faction;
}

export interface BuildFactoryAction extends Action {
  type: 'buildFactory';
  center: Tile;
  water: number;
  metal: number;
}

export interface WaitAction extends Action {
  type: 'wait';
}

export interface BuildRobotAction extends Action {
  type: 'buildRobot';
  robotType: RobotType;
}

export interface WaterAction extends Action {
  type: 'water';
}

export interface RepeatableAction extends Action {
  repeat: number;
  n?: number;
}

export interface MoveAction extends RepeatableAction {
  type: 'move';
  direction: Direction;
}

export interface TransferAction extends RepeatableAction {
  type: 'transfer';
  direction: Direction;
  resource: Resource;
  amount: number;
}

export interface PickupAction extends RepeatableAction {
  type: 'pickup';
  resource: Resource;
  amount: number;
}

export interface DigAction extends RepeatableAction {
  type: 'dig';
}

export interface SelfDestructAction extends RepeatableAction {
  type: 'selfDestruct';
}

export interface RechargeAction extends RepeatableAction {
  type: 'recharge';
  targetPower: number;
}

export type SetupAction = BidAction | BuildFactoryAction | WaitAction;
export type FactoryAction = BuildRobotAction | WaterAction;
export type RobotAction = MoveAction | TransferAction | PickupAction | DigAction | SelfDestructAction | RechargeAction;

export interface Board {
  energy: number[][];
  tileType: number[][];
  relicNodes: number[][];
  relicNodeConfigs: number[][][];
  // ice: number[][];
  // lichen: number[][];
  // strains: number[][];
}

export interface Unit {
  unitId: string;

  tile: Tile;

  // power: number;
  // cargo: Cargo;
}

// export interface Factory extends Unit {
//   strain: number;
//   action: FactoryAction | null;

//   lichen: number;
// }

export interface Robot extends Unit {
  // type: RobotType;
  // actionQueue: RobotAction[];
  // position: number[];
  energy: number;
}

export interface Team {
  name: string;
  points: number;
  // faction: Faction;

  // water: number;
  // metal: number;

  // factories: Factory[];
  robots: Robot[];

  // strains: Set<number>;

  // placeFirst: boolean;
  // factoriesToPlace: number;

  // action: SetupAction | null;

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
  unit_sap_drain: number;
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
