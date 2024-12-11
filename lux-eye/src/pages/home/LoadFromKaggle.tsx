import { Text } from '@mantine/core';
import { HomeCard } from './HomeCard';

export function LoadFromKaggle(): JSX.Element {
  return (
    <HomeCard title="Load from Kaggle">
      {/* prettier-ignore */}
      <Text>
        Episodes can be loaded straight from Kaggle notebooks.
        See the <a href="https://www.kaggle.com/code/jmerle/lux-eye-s3-notebook-integration" target="_blank" rel="noreferrer">example notebook</a> for instructions.
      </Text>
    </HomeCard>
  );
}
