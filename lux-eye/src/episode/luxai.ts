import {
  Board,
  Episode,
  EpisodeMetadata,
  Faction,
  FactoryAction,
  Robot,
  RobotAction,
  RobotType,
  SetupAction,
  Step,
  Team,
} from './model';

function transpose<T>(matrix: T[][]): T[][] {
  return matrix[0].map((value, i) => matrix.map(row => row[i]));
}

function isSetupAction(data: any): boolean {
  if (data === null) {
    return false;
  }

  return (
    (data.bid !== undefined && data.faction !== undefined) ||
    (data.spawn !== undefined && data.water !== undefined && data.metal !== undefined) ||
    Object.keys(data).length === 0
  );
}

function parseSetupAction(data: any): SetupAction {
  if (data.bid !== undefined && data.faction !== undefined) {
    return {
      type: 'bid',
      bid: data.bid,
      faction: data.faction,
    };
  } else if (data.spawn !== undefined && data.water !== undefined && data.metal !== undefined) {
    return {
      type: 'buildFactory',
      center: {
        x: data.spawn[0],
        y: data.spawn[1],
      },
      water: data.water,
      metal: data.metal,
    };
  } else if (Object.keys(data).length === 0) {
    return {
      type: 'wait',
    };
  } else {
    throw new Error(`Cannot parse '${data}' as setup action`);
  }
}

function parseFactoryAction(data: any): FactoryAction {
  if (data === 0) {
    return {
      type: 'buildRobot',
      robotType: RobotType.Light,
    };
  } else if (data === 1) {
    return {
      type: 'buildRobot',
      robotType: RobotType.Heavy,
    };
  } else if (data === 2) {
    return {
      type: 'water',
    };
  } else {
    throw new Error(`Cannot parse '${data}' as factory action`);
  }
}

function parseRobotAction(data: any): RobotAction {
  switch (data[0]) {
    case 0:
      return {
        type: 'move',
        repeat: data[4],
        n: data[5],
        direction: data[1],
      };
    case 1:
      return {
        type: 'transfer',
        repeat: data[4],
        n: data[5],
        direction: data[1],
        resource: data[2],
        amount: data[3],
      };
    case 2:
      return {
        type: 'pickup',
        repeat: data[4],
        n: data[5],
        resource: data[2],
        amount: data[3],
      };
    case 3:
      return {
        type: 'dig',
        repeat: data[4],
        n: data[5],
      };
    case 4:
      return {
        type: 'selfDestruct',
        repeat: data[4],
        n: data[5],
      };
    case 5:
      return {
        type: 'recharge',
        repeat: data[4],
        n: data[5],
        targetPower: data[3],
      };
    default:
      throw new Error(`Cannot parse '${data}' as robot action`);
  }
}

export function isLuxAISEpisode(data: any): boolean {
  // console.log(data);
  // TODO: chekc this
  return typeof data === 'object';
  // return typeof data === 'object' && data.observations !== undefined && data.actions !== undefined;
}

export function parseLuxAISEpisode(data: any, extra: Partial<EpisodeMetadata> = {}): Episode {
  console.log('PARSING');
  // console.log(data);
  let metadata: EpisodeMetadata = { teamNames: ['Player A', 'Player B'], seed: undefined };
  metadata = {
    ...metadata,
    ...extra,
  };
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

    if (data.observations.length === data.actions.length) {
      if (i < data.actions.length - 1) {
        actions = data.actions[i + 1];
      }
    } else if (i < data.actions.length) {
      actions = data.actions[i];
    }

    const board: Board = {
      energy: transpose(obs.map_features.energy),
      tileType: transpose(obs.map_features.tile_type),
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
      const robots: Robot[] = [];
      // TODO: might not use a mask in the future.
      for (let unitIdx = 0; unitIdx < obs.units_mask.length; unitIdx++) {
        // const rawRobot = obs.units[unit_idx];
        robots.push({
          unitId: `unit_${unitIdx}`,
          tile: {
            x: obs.units.position[j][unitIdx][0],
            y: obs.units.position[j][unitIdx][1],
          },
          energy: obs.units.energy[j][unitIdx],
        });
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
        points: 0,
        error: error,
        //   name: metadata.teamNames[j],
        //   faction: rawTeam.faction,

        //   water: rawTeam.water,
        //   metal: rawTeam.metal,

        //   factories,
        robots,

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

  return { steps, metadata };
}
