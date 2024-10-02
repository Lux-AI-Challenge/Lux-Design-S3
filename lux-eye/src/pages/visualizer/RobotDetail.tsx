import { Grid, Text, Tooltip } from '@mantine/core';
import { Direction, Robot, RobotAction } from '../../episode/model';
import { UnitCard } from './UnitCard';
import { useStore } from '../../store';

interface RobotDetailProps {
  robot: Robot;
}

function formatAction(action: RobotAction): string {
  switch (action.type) {
    case 'move': {
      const direction = Direction[action.direction].toLowerCase();
      return `Move ${direction}`;
    }
    case 'sap': {
      const target = action.target ? `(${action.target.x}, ${action.target.y})` : '';
      if (action.validSap) {
        return `Sap ${target}`;
      } else {
        return `Sap ${target} (Out of range)`;
      }
    }
  }
}

export function RobotDetail({ robot }: RobotDetailProps): JSX.Element {
  const episode = useStore(state => state.episode);
  return (
    <UnitCard tiles={[robot.tile]} tileToSelect={robot.tile}>
      <Grid gutter={0}>
        <Grid.Col span={4}>
          <Text size="sm">
            <b>{robot.unitId}</b>
          </Text>
        </Grid.Col>
        <Grid.Col span={8}>
          <Text size="sm">
            Tile: ({robot.tile.x}, {robot.tile.y})
          </Text>
        </Grid.Col>
        <Grid.Col span={4}>
          <Text size="sm">Energy: {robot.energy}</Text>
        </Grid.Col>
        {(
          <Grid.Col span={8}>
            {robot.prevAction && <Text size="sm">Previous Action: {formatAction(robot.prevAction)}</Text>}
            {robot.action && <Text size="sm">Next Action: {formatAction(robot.action)}</Text>}
          </Grid.Col>
        )}
      </Grid>
    </UnitCard>
  );
}
