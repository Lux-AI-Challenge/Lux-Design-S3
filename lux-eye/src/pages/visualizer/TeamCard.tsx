import { Badge, Grid, MantineShadow, Paper, Space, Tabs, Title } from '@mantine/core';
import { IconCrown } from '@tabler/icons';
import { useCallback, useMemo } from 'react';
import { Episode, Unit } from '../../episode/model';
import { useStore } from '../../store';
import { getTeamColor } from '../../utils/colors';
import { RobotDetail } from './RobotDetail';
import { UnitList } from './UnitList';
import { funcPoints } from './VisualizerPage';

export function getWinnerInfo(episode: Episode, team: number): [won: boolean, reason: string | null] {
  const lastStep = episode.steps[episode.steps.length - 1];

  const me = lastStep.teams[team];
  const opponent = lastStep.teams[team === 0 ? 1 : 0];

  const meError = episode.steps.map(step => step.teams[team].error).some(error => error !== null);
  const opponentError = episode.steps.map(step => step.teams[team === 0 ? 1 : 0].error).some(error => error !== null);

  const meWins = me.wins;
  const opponentWins = opponent.wins;
  if (meError && opponentError) {
    return [true, 'Draw, both teams errored'];
  } else if (meError && !opponentError) {
    return [false, null];
  } else if (!meError && opponentError) {
    return [true, 'Winner by opponent error'];
  } else if (lastStep.step === (episode.params.max_steps_in_match + 1) * episode.params.match_count_per_episode) {
    if (meWins > opponentWins) {
      return [true, 'Winner by total match wins'];
    } else if (meWins === opponentWins) {
      return [true, 'Draw???'];
    } else {
      return [false, null];
    }
  } else {
    return [true, 'Draw, game ended prematurely'];
  }
}

function compareUnits(a: Unit, b: Unit): number {
  const partsA = a.unitId.split('_');
  const partsB = b.unitId.split('_');

  if (partsA[0] === partsB[0]) {
    return parseInt(partsA[1]) - parseInt(partsB[1]);
  }

  return partsA[0].localeCompare(partsB[0]);
}

interface TeamCardProps {
  id: number;
  tabHeight: number;
  shadow?: MantineShadow;
}

export function TeamCard({ id, tabHeight, shadow }: TeamCardProps): JSX.Element {
  const episode = useStore(state => state.episode)!;
  const turn = useStore(state => state.turn);

  const step = episode.steps[turn];
  const team = step.teams[id];

  const [isWinner, winnerReason] = getWinnerInfo(episode, id);
  const sortedRobots = useMemo(() => team.robots.sort(compareUnits), [team]);
  const robotRenderer = useCallback((index: number) => <RobotDetail robot={sortedRobots[index]} />, [sortedRobots]);
  const robotTileGetter = useCallback((index: number) => [sortedRobots[index].tile], [sortedRobots]);

  tabHeight = step.step < 0 ? tabHeight - 100 : tabHeight;
  tabHeight = isWinner ? tabHeight - 24 : tabHeight;

  return (
    <Paper shadow={shadow} p="md" withBorder>
      <Title order={3} style={{ color: getTeamColor(id, 1.0) }}>
        {isWinner && <IconCrown style={{ verticalAlign: 'middle', marginRight: '2px' }} />}
        {team.name}
      </Title>

      <Badge color="dark">{team.name}</Badge>
      {isWinner && (
        <Badge color={id === 0 ? 'blue' : 'red'} ml={8}>
          {winnerReason}
        </Badge>
      )}

      <Space h="xs" />

      <Grid columns={2} gutter={0}>
        <Grid.Col span={1}>
          <b>Ships:</b> {sortedRobots.length}
        </Grid.Col>
        <Grid.Col span={1}>
          <b>Points:</b> {team.points}
        </Grid.Col>
        <Grid.Col span={1}>
          <b>Wins:</b> {team.wins}
        </Grid.Col>
        {team.error && (
          <Grid.Col span={2}>
            <b>Error:</b> {team.error}
          </Grid.Col>
        )}
      </Grid>

      <Space h="xs" />

      <Tabs defaultValue="robots" keepMounted={false} color={id === 0 ? 'blue' : 'red'}>
        <Tabs.List mb="xs" grow>
          <Tabs.Tab value="robots">Robots</Tabs.Tab>
        </Tabs.List>
        <Tabs.Panel value="robots">
          <UnitList
            name="robots"
            height={tabHeight}
            itemCount={sortedRobots.length}
            itemRenderer={robotRenderer}
            tileGetter={robotTileGetter}
          />
        </Tabs.Panel>
      </Tabs>
    </Paper>
  );
}
