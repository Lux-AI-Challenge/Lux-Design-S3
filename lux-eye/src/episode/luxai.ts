import {
  Board,
  Direction,
  EnvParams,
  Episode,
  EpisodeMetadata,
  Robot,
  RobotAction,
  Step,
  Team,
  Tile,
  TileType,
} from './model';

function transpose<T>(matrix: T[][]): T[][] {
  return matrix[0].map((value, i) => matrix.map(row => row[i]));
}
function parseRobotAction(params: EnvParams, robotPosition: Tile, data: any): RobotAction {
  switch (data[0]) {
    case 0:
      return {
        type: 'move',
        direction: Direction.Center,
      };
      case 1:
        return {
          type: 'move',
          direction: Direction.Up,
        };
      case 2:
        return {
          type: 'move',
          direction: Direction.Right,
        };
      case 3:
        return {
          type: 'move',
        direction: Direction.Down,
      };
      case 4:
        return {
          type: 'move',
          direction: Direction.Left,
        };
      case 5:
        const target = {
          x: robotPosition.x + data[1],
          y: robotPosition.y + data[2],
        }
        return {
          type: 'sap',
          target,
          validSap: Math.max(Math.abs(data[1]), Math.abs(data[2])) <= params.unit_sap_range && target.x >= 0 && target.x < params.map_width && target.y >= 0 && target.y < params.map_height,
        };
    default:
      throw new Error(`Cannot parse '${data}' as robot action`);
  }
}

export function isLuxAISEpisode(data: any): boolean {
  return typeof data === 'object' && data.observations !== undefined && data.actions !== undefined;
}

export function parseLuxAISEpisode(data: any, extra: Partial<EpisodeMetadata> = {}): Episode {
  let metadata: EpisodeMetadata = { teamNames: ['Player A', 'Player B'], seed: undefined };
  metadata = {
    ...metadata,
    ...extra,
  };
  const params: EnvParams = data.params;
  if (data.metadata) {
    if (data.metadata['players']) {
      for (let i = 0; i < 2; i++) {
        metadata.teamNames[i] = data.metadata['players'][`player_${i}`];
      }
    }
    if (data.metadata['seed']) {
      metadata.seed = data.metadata['seed'];
    }
  }

  const steps: Step[] = [];

  for (let i = 0; i < data.observations.length; i++) {
    const obs = data.observations[i];

    let actions: Record<string, any> = {
      // eslint-disable-next-line @typescript-eslint/naming-convention
      player_0: {},
      // eslint-disable-next-line @typescript-eslint/naming-convention
      player_1: {},
    };
    let prevActions = { ...actions };

    if (data.observations.length === data.actions.length) {
      if (i < data.actions.length - 1) {
        actions = data.actions[i + 1];
      }
    } else if (i < data.actions.length) {
      actions = data.actions[i];
      if (i > 0) {
        prevActions = data.actions[i - 1];
      }
    }
    const board: Board = {
      energy: obs.map_features.energy,
      energyNodes: obs.energy_nodes,
      tileType: obs.map_features.tile_type,
      relicNodes: obs.relic_nodes,
      relicNodeConfigs: obs.relic_node_configs,
      visionPowerMap: obs.vision_power_map,
    };

    const teams: Team[] = [];
    for (let j = 0; j < 2; j++) {
      const error: string | null = null;
      const sensorMask = obs.vision_power_map[j].map((map: number[]) => map.map((value: number) => value > 0));
      const robots: Robot[] = [];
      for (let unitIdx = 0; unitIdx < obs.units_mask[j].length; unitIdx++) {
        if (obs.units_mask[j][unitIdx]) {
          const robotPosition = {
            x: obs.units.position[j][unitIdx][0],
            y: obs.units.position[j][unitIdx][1],
          }
          const robotAction = actions[`player_${j}`][unitIdx];
          robots.push({
            unitId: `unit_${unitIdx}`,
            tile: robotPosition,
            energy: parseInt(obs.units.energy[j][unitIdx]),
            action: robotAction ? parseRobotAction(params, robotPosition, robotAction) : null,
            prevAction: prevActions[`player_${j}`][unitIdx] ? parseRobotAction(params, robotPosition, prevActions[`player_${j}`][unitIdx]) : null,
          });
        }
      }
      teams.push({
        name: metadata.teamNames[j],
        points: obs.team_points[j],
        wins: obs.team_wins[j],
        error: error,
        robots,

        sensorMask,
      });
    }
    steps.push({
      step: obs.steps,
      board,
      teams: teams as [Team, Team],
    });
  }
  return { steps, metadata, params: params };
}

export function getMatchIdx(step: number, envParams: EnvParams): number {
  return Math.floor((step-1) / (envParams.max_steps_in_match + 1));
}

export function parseTileType(tileType: number): string {
  switch (tileType) {
    case 0:
      return 'Space';
    case 1:
      return 'Nebula';
    case 2:
      return 'Asteroid';
  }
  return 'Null';
}
