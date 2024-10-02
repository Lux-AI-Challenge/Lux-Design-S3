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
  // TODO: check this
  return typeof data === 'object';
  // return typeof data === 'object' && data.observations !== undefined && data.actions !== undefined;
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
      // ice: transpose(obs.map_features.ice),
      // lichen: transpose(obs.board.lichen),
      // strains: transpose(obs.board.lichen_strains),
    };
    // if (i === 0) {
    //   board = {
    //     energy: transpose(obs.map_features.energy),
    //     tile_type: transpose(obs.map_features.tile_type),
    //     // ice: transpose(obs.map_features.ice),
    //     // lichen: transpose(obs.board.lichen),
    //     // strains: transpose(obs.board.lichen_strains),
    //   };
    // } else if (Array.isArray(obs.board.rubble)) {
    //   board = {
    //     rubble: transpose(obs.board.rubble),
    //     ore: JSON.parse(JSON.stringify(steps[i - 1].board.ore)),
    //     ice: JSON.parse(JSON.stringify(steps[i - 1].board.ice)),
    //     lichen: transpose(obs.board.lichen),
    //     strains: transpose(obs.board.lichen_strains),
    //   };
    // } else {
    //   board = JSON.parse(JSON.stringify(steps[i - 1].board));

    //   for (const [item, grid] of <[string, number[][]][]>[
    //     ['rubble', board.rubble],
    //     ['lichen', board.lichen],
    //     ['lichen_strains', board.strains],
    //   ]) {
    //     for (const key in obs.board[item]) {
    //       const [x, y] = key.split(',').map(part => parseInt(part));
    //       grid[y][x] = obs.board[item][key];
    //     }
    //   }
    // }

    const teams: Team[] = [];
    for (let j = 0; j < 2; j++) {
      // const playerId = `player_${j}`;
      const error: string | null = null;

      // if (obs.teams[playerId] === undefined) {
      //   const rawPlayer =
      //     data.observations[1].teams[playerId] !== undefined ? data.observations[1].teams[playerId] : null;

      //   teams.push({
      //     name: metadata.teamNames[j],
      //     faction: rawPlayer !== null ? rawPlayer.faction : Faction.None,

      //     water: 0,
      //     metal: 0,

      //     factories: [],
      //     robots: [],

      //     strains: new Set(),

      //     placeFirst: rawPlayer !== null ? rawPlayer.place_first : false,
      //     factoriesToPlace: rawPlayer !== null ? rawPlayer.factories_to_place : 0,

      //     action: actions[playerId] !== null ? parseSetupAction(actions[playerId]) : null,

      //     error,
      //   });

      //   continue;
      // }

      // const factories: Factory[] = [];
      // for (const unitId of Object.keys(obs.factories[playerId])) {
      //   const rawFactory = obs.factories[playerId][unitId];

      //   let lichen = 0;
      //   for (let y = 0; y < board.lichen.length; y++) {
      //     for (let x = 0; x < board.lichen.length; x++) {
      //       if (board.strains[y][x] === rawFactory.strain_id) {
      //         lichen += board.lichen[y][x];
      //       }
      //     }
      //   }

      //   if (actions[playerId] === null) {
      //     error = 'Actions object is null';
      //   }

      //   factories.push({
      //     unitId,

      //     tile: {
      //       x: rawFactory.pos[0],
      //       y: rawFactory.pos[1],
      //     },

      //     power: rawFactory.power,
      //     cargo: rawFactory.cargo,

      //     strain: rawFactory.strain_id,
      //     action:
      //       actions[playerId] !== null && actions[playerId][unitId] !== undefined
      //         ? parseFactoryAction(actions[playerId][unitId])
      //         : null,

      //     lichen,
      //   });
      // }
      // console.log(obs);
      //sensorMask
      const sensorMask = obs.sensor_mask[j];
      const robots: Robot[] = [];
      // TODO: might not use a mask in the future.
      // console.log(obs.units_mask);
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
      // for (const unitId of Object.keys(obs.units[playerId])) {
      //   const rawRobot = obs.units[playerId][unitId];
      //   // const actionQueue =
      //   //   actions[playerId] !== null && actions[playerId][unitId] !== undefined
      //   //     ? actions[playerId][unitId]
      //   //     : rawRobot.action_queue;

      //   // if (actions[playerId] === null) {
      //   //   error = 'Actions object is null';
      //   // }

      //   robots.push({
      //     unitId,

      //     tile: {
      //       x: rawRobot.position[0],
      //       y: rawRobot.position[1],
      //     },
      //     energy: 0
      //   });
      // }

      // const rawTeam = obs.teams[playerId];
      teams.push({
        name: metadata.teamNames[j],
        points: obs.team_points[j],
        error: error,
        //   name: metadata.teamNames[j],
        //   faction: rawTeam.faction,

        //   water: rawTeam.water,
        //   metal: rawTeam.metal,

        //   factories,
        robots,

        sensorMask,
        //   strains: new Set(rawTeam.factory_strains),

        //   placeFirst: rawTeam.place_first,
        //   factoriesToPlace: rawTeam.factories_to_place,

        //   action: isSetupAction(actions[playerId]) ? parseSetupAction(actions[playerId]) : null,

        //   error,
      });
    }
    steps.push({
      step: obs.steps,
      board,
      teams: teams as [Team, Team],
    });
  }
  // console.log(steps);
  return { steps, metadata, params: params };
}

export function getMatchIdx(step: number, envParams: EnvParams): number {
  return Math.floor(step / envParams.max_steps_in_match);
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
