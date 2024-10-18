import { parseLuxAISEpisode } from './luxai';
import { EnvParams, Episode, EpisodeMetadata } from './model';

export function isKaggleEnvironmentsEpisode(data: any): boolean {
  return typeof data === 'object' && data.steps !== undefined;
}

export function parseKaggleEnvironmentsEpisode(data: any): Episode {
  const observations = [];
  const actions = [];
  const extra: Partial<EpisodeMetadata> = {};
  if (typeof data.info === 'object' && data.info.TeamNames !== undefined) {
    extra.teamNames = data.info.TeamNames;
  }
  if (typeof data.configuration == 'object' && data.configuration.seed !== undefined) {
    extra.seed = data.configuration.seed;
  }
  let params: Partial<EnvParams> = {}
  for (const step of data.steps) {
    if (step[0].info.replay) {
      const obs = step[0].info.replay.observations[0]

      observations.push(obs);
      if (step[0].info.replay.actions) {
        actions.push(step[0].info.replay.actions[0])
      }
      if (step[0].info.replay.params) {
        params = step[0].info.replay.params
      }
  }
}

  return parseLuxAISEpisode({ observations, actions, params }, extra);
}
