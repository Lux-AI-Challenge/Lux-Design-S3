import { useHover, useMergedRef, useMouse } from '@mantine/hooks';
import { useCallback, useEffect, useState } from 'react';
import { EnvParams, Robot, Step, Tile } from '../../episode/model';
import { useStore } from '../../store';
import { getTeamColor } from '../../utils/colors';

interface SizeConfig {
  gutterSize: number;
  tileSize: number;
  boardSize: number;
  tilesPerSide: number;
}
export interface DisplayConfig {
  energyField: boolean;
  sensorMask: boolean;
  relicConfigs: boolean;
}
interface ThemeConfig {
  minimalTheme: boolean;
}

type Config = SizeConfig & ThemeConfig & DisplayConfig;

function getSizeConfig(maxWidth: number, step: Step): SizeConfig {
  const gutterSize = 1;
  const tilesPerSide = step.board.energy.length;

  let tileSize = Math.floor(Math.sqrt(maxWidth));
  let boardSize = tileSize * tilesPerSide + gutterSize * (tilesPerSide + 1);

  while (boardSize > maxWidth) {
    tileSize--;
    boardSize -= tilesPerSide;
  }

  return {
    gutterSize,
    tileSize,
    boardSize,
    tilesPerSide,
  };
}

function tileToCanvas(sizes: SizeConfig, tile: Tile): [number, number] {
  return [
    (tile.x + 1) * sizes.gutterSize + tile.x * sizes.tileSize,
    (tile.y + 1) * sizes.gutterSize + tile.y * sizes.tileSize,
  ];
}

// function scale(value: number, relativeMin: number, relativeMax: number): number {
//   const clampedValue = Math.max(Math.min(value, relativeMax), relativeMin);
//   return (clampedValue - relativeMin) / (relativeMax - relativeMin);
// }

function drawTileBackgrounds(ctx: CanvasRenderingContext2D, config: Config, step: Step, envParams: EnvParams): void {
  const board = step.board;

  for (let tileY = 0; tileY < config.tilesPerSide; tileY++) {
    for (let tileX = 0; tileX < config.tilesPerSide; tileX++) {
      const [canvasX, canvasY] = tileToCanvas(config, { x: tileX, y: tileY });

      ctx.fillStyle = 'white';
      ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);

      let color: string;
      if (board.tileType[tileX][tileY] == 1) {
        color = '#5B5F97';
      } else if (board.tileType[tileX][tileY] == 2) {
        color = '#2c3e50';
      } else {
        const rgb = 230;
        // const base = isDay ? 0.1 : 0.2;
        color = `rgba(${rgb}, ${rgb}, ${rgb}, 1)`;
      }

      ctx.fillStyle = color;
      ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);

      if (config.sensorMask) {
        for (const team of step.teams) {
          if (team.sensorMask[tileX][tileY]) {
            ctx.fillStyle = `rgba(255, 0, 0, 0.1)`;
            ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);
          }
        }
      }
      if (config.energyField) {
        const energy = board.energy[tileX][tileY];
        if (energy > 0) {
          ctx.fillStyle = `rgba(0, 255, 0, ${energy / 50})`;
          ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);
        } else {
          ctx.fillStyle = `rgba(255, 0, 0, ${-energy / 50})`;
          ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);
        }
      }
    }
  }

  // for (let i = 0; i < board.energyNodes.length; i++) {
  //   const [canvasX, canvasY] = tileToCanvas(config, { x: board.energyNodes[i][0], y: board.energyNodes[i][1] });
  //   ctx.fillStyle = 'green';
  //   ctx.fillRect(canvasX, canvasY, config.tileSize / 2, config.tileSize / 2);
  // }

  for (let i = 0; i < board.relicNodes.length; i++) {
    const [canvasX, canvasY] = tileToCanvas(config, { x: board.relicNodes[i][0], y: board.relicNodes[i][1] });

    ctx.fillStyle = 'orange';
    ctx.fillRect(canvasX + config.tileSize / 4, canvasY + config.tileSize / 4, config.tileSize / 2, config.tileSize / 2);
    if (config.relicConfigs) {
      for (
        let dx = -Math.floor(envParams.relic_config_size / 2);
        dx < Math.ceil(envParams.relic_config_size / 2);
        dx++
      ) {
        for (
          let dy = -Math.floor(envParams.relic_config_size / 2);
          dy < Math.ceil(envParams.relic_config_size / 2);
          dy++
        ) {
          const nx = board.relicNodes[i][0] + dx;
          const ny = board.relicNodes[i][1] + dy;
          if (nx < 0 || nx >= config.tilesPerSide || ny < 0 || ny >= config.tilesPerSide) {
            continue;
          }
          if (
            board.relicNodeConfigs[i][dx + Math.floor(envParams.relic_config_size / 2)][
              dy + Math.floor(envParams.relic_config_size / 2)
            ] != 0
          ) {
            const [canvasX, canvasY] = tileToCanvas(config, { x: nx, y: ny });
            ctx.fillStyle = 'rgba(100, 100, 0, 0.1)';
            ctx.fillRect(canvasX, canvasY, config.tileSize, config.tileSize);
          }
        }
      }
    }
  }

  ctx.restore();
}

function drawRobot(
  ctx: CanvasRenderingContext2D,
  config: Config,
  robot: Robot,
  team: number,
  selectedTile: Tile | null,
): void {
  const [canvasX, canvasY] = tileToCanvas(config, robot.tile);

  const isSelected = selectedTile !== null && robot.tile.x === selectedTile.x && robot.tile.y === selectedTile.y;

  ctx.fillStyle = getTeamColor(team, 1.0);
  ctx.strokeStyle = 'black';
  ctx.lineWidth = isSelected ? 2 : 1;

  const radius = config.tileSize / 2 - 3;

  ctx.beginPath();
  ctx.arc(canvasX + config.tileSize / 2, canvasY + config.tileSize / 2, radius, 0, 2 * Math.PI);
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  if (robot.prevAction && robot.prevAction.type === 'sap' && robot.prevAction.validSap) {
    const [canvasX, canvasY] = tileToCanvas(config, robot.tile);
    const [targetCanvasX, targetCanvasY] = tileToCanvas(config, robot.prevAction.target);

    ctx.strokeStyle = getTeamColor(team, 0.5); // Semi opaque line
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(canvasX + config.tileSize / 2, canvasY + config.tileSize / 2);
    ctx.lineTo(targetCanvasX + config.tileSize / 2, targetCanvasY + config.tileSize / 2);
    ctx.stroke();
  }
}

function drawSelectedTile(ctx: CanvasRenderingContext2D, config: Config, selectedTile: Tile): void {
  const [canvasX, canvasY] = tileToCanvas(config, selectedTile);

  ctx.fillStyle = 'black';

  ctx.fillRect(
    canvasX - config.gutterSize,
    canvasY - config.gutterSize,
    config.tileSize + config.gutterSize * 2,
    config.gutterSize,
  );

  ctx.fillRect(
    canvasX - config.gutterSize,
    canvasY + config.tileSize,
    config.tileSize + config.gutterSize * 2,
    config.gutterSize,
  );

  ctx.fillRect(
    canvasX - config.gutterSize,
    canvasY - config.gutterSize,
    config.gutterSize,
    config.tileSize + config.gutterSize * 2,
  );

  ctx.fillRect(
    canvasX + config.tileSize,
    canvasY - config.gutterSize,
    config.gutterSize,
    config.tileSize + config.gutterSize * 2,
  );

  ctx.restore();
}

function drawBoard(
  ctx: CanvasRenderingContext2D,
  config: Config,
  step: Step,
  envParams: EnvParams,
  selectedTile: Tile | null,
): void {
  ctx.save();

  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, config.boardSize, config.boardSize);
  ctx.restore();

  drawTileBackgrounds(ctx, config, step, envParams);
  const robotPositions = new Map<string, number>();

  for (let i = 0; i < 2; i++) {
    for (const robot of step.teams[i].robots) {
      const key = `${i},${robot.tile.x},${robot.tile.y}`;
      if (robotPositions.has(key)) {
        robotPositions.set(key, (robotPositions.get(key) ?? 0) + 1);
      } else {
        robotPositions.set(key, 1);
      }
    }
  }
  for (let i = 0; i < 2; i++) {
    for (const robot of step.teams[i].robots) {
      drawRobot(ctx, config, robot, i, selectedTile);
    }
  }
  robotPositions.forEach((count, key) => {
    const [team, x, y] = key.split(',').map(Number);
    const [canvasX, canvasY] = tileToCanvas(config, { x, y });
    ctx.fillStyle = 'white';
    ctx.font = 'bold 10px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(count.toString(), canvasX + config.tileSize / 2, canvasY + config.tileSize / 2);
  });

  if (selectedTile !== null) {
    drawSelectedTile(ctx, config, selectedTile);
  }
}

interface BoardProps {
  maxWidth: number;
}

export function Board({ maxWidth }: BoardProps): JSX.Element {
  const { ref: canvasMouseRef, x: mouseX, y: mouseY } = useMouse<HTMLCanvasElement>();
  const { ref: canvasHoverRef, hovered } = useHover<HTMLCanvasElement>();
  const canvasRef = useMergedRef(canvasMouseRef, canvasHoverRef);

  const episode = useStore(state => state.episode);
  const turn = useStore(state => state.turn);
  const displayConfig = useStore(state => state.displayConfig);
  const selectedTile = useStore(state => state.selectedTile);
  const setSelectedTile = useStore(state => state.setSelectedTile);

  const minimalTheme = useStore(state => state.minimalTheme);

  const [sizeConfig, setSizeConfig] = useState<SizeConfig>({
    gutterSize: 0,
    tileSize: 0,
    boardSize: 0,
    tilesPerSide: 0,
  });

  const step = episode!.steps[turn];
  const envParams = episode!.params;

  const onMouseLeave = useCallback(() => {
    setSelectedTile(null, true);
  }, []);

  useEffect(() => {
    const newSizeConfig = getSizeConfig(maxWidth, step);
    if (
      newSizeConfig.gutterSize !== sizeConfig.gutterSize ||
      newSizeConfig.tileSize !== sizeConfig.tileSize ||
      newSizeConfig.boardSize !== sizeConfig.boardSize ||
      newSizeConfig.tilesPerSide !== sizeConfig.tilesPerSide
    ) {
      setSizeConfig(newSizeConfig);
    }
  }, [maxWidth, episode]);

  useEffect(() => {
    if (!hovered) {
      return;
    }

    for (let tileY = 0; tileY < sizeConfig.tilesPerSide; tileY++) {
      for (let tileX = 0; tileX < sizeConfig.tilesPerSide; tileX++) {
        const tile = { x: tileX, y: tileY };
        const [canvasX, canvasY] = tileToCanvas(sizeConfig, tile);

        if (
          mouseX >= canvasX &&
          mouseX < canvasX + sizeConfig.tileSize &&
          mouseY >= canvasY &&
          mouseY < canvasY + sizeConfig.tileSize
        ) {
          setSelectedTile(tile, true);
          return;
        }
      }
    }
  }, [sizeConfig, mouseX, mouseY, hovered]);

  useEffect(() => {
    if (sizeConfig.tileSize <= 0) {
      return;
    }

    const ctx = canvasMouseRef.current.getContext('2d')!;

    const config = {
      ...sizeConfig,
      minimalTheme,
      ...displayConfig,
    };

    drawBoard(ctx, config, step, envParams, selectedTile);
  }, [step, envParams, sizeConfig, selectedTile, minimalTheme, displayConfig]);

  return (
    <canvas ref={canvasRef} width={sizeConfig.boardSize} height={sizeConfig.boardSize} onMouseLeave={onMouseLeave} />
  );
}
