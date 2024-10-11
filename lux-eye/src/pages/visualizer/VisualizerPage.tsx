import { Center, createStyles, Grid, MediaQuery, Paper, Stack } from '@mantine/core';
import { useElementSize } from '@mantine/hooks';
import { Navigate, useLocation } from 'react-router-dom';
import { useStore } from '../../store';
import { Board } from './Board';
import { Chart, ChartFunction } from './Chart';
import { TeamCard } from './TeamCard';
import { TurnControl } from './TurnControl';

const useStyles = createStyles(theme => ({
  container: {
    margin: '0 auto',
    width: '1500px',

    [theme.fn.smallerThan(1500)]: {
      width: '100%',
    },
  },
}));

// function funcCargo(unitType: 'factories' | 'robots', resource: keyof Cargo): ChartFunction {
//   return team => (team[unitType] as Unit[]).reduce((acc, val) => acc + val.cargo[resource], 0);
// }

export const funcPoints: ChartFunction = (team, board) => {
  board;
  return team.points;
};

export const funcTotalUnitEnergy: ChartFunction = (team, board) => {
  board;
  return team.robots.reduce((acc, val) => acc + val.energy, 0);
};
export function VisualizerPage(): JSX.Element {
  const { classes } = useStyles();

  const episode = useStore(state => state.episode);

  const { search } = useLocation();

  const { ref: boardContainerRef, width: maxBoardWidth } = useElementSize();

  if (episode === null) {
    return <Navigate to={`/${search}`} />;
  }

  const teamCards = [];
  for (let i = 0; i < 2; i++) {
    teamCards.push(<TeamCard id={i} tabHeight={570} shadow="xs" />);
  }

  return (
    <div className={classes.container}>
      <Grid columns={24}>
        <MediaQuery smallerThan="md" styles={{ display: 'none' }}>
          <Grid.Col span={7}>{teamCards[0]}</Grid.Col>
        </MediaQuery>
        <Grid.Col span={24} md={10}>
          <Paper shadow="xs" p="md" withBorder>
            <Stack>
              <Center ref={boardContainerRef}>
                <Board maxWidth={maxBoardWidth} />
              </Center>
              <TurnControl showHotkeysButton={true} showOpenButton={false} />
            </Stack>
          </Paper>
        </Grid.Col>
        <MediaQuery largerThan="md" styles={{ display: 'none' }}>
          <Grid.Col span={24}>{teamCards[0]}</Grid.Col>
        </MediaQuery>
        <Grid.Col span={24} md={7}>
          {teamCards[1]}
        </Grid.Col>
      </Grid>
      <Grid columns={12}>
        <Grid.Col span={12} md={6}>
          <Chart title="Points" func={funcPoints} />
        </Grid.Col>
        <Grid.Col span={12} md={6}>
          <Chart title="Total Unit Energy" func={funcTotalUnitEnergy} />
        </Grid.Col>
      </Grid>
    </div>
  );
}
