# Lux-Design S3

Welcome to the Lux AI Challenge Season 3, an official NeurIPS 2024 competition!

The Lux AI Challenge is a competition where competitors design agents to tackle a multi-variable optimization, resource gathering, and allocation problem in a 1v1 scenario against other competitors. In addition to optimization, successful agents must be capable of analyzing their opponents and developing appropriate policies to get the upper hand.

Unique to this season is the introduction of partial observability and meta-learning style competition. A game against an opponent is won when the team wins a best of 5 match series where each match uses the same randomized map and game parameters.

The full game rules/specs can be found [here](docs/specs.md).

## Getting Started

We recommend using a python package manager like conda/mamba to install the dependencies.

```bash
mamba create -n "lux-s3" "python==3.11"
git clone https://github.com/Lux-AI-Challenge/Lux-Design-S3/
pip install -e Lux-Design-S3/src
```

To verify your installation, you can run a match between two random agents:

```bash
luxai-s3 --help
```

```bash
luxai-s3 path/to/bot/main.py path/to/bot/main.py --output replay.json
```

Then upload the replay.json to the online visualizer here: https://lux-eye-s3.netlify.app/ (a link on the lux-ai.org website will be up soon) 
<!-- https://s3vis.lux-ai.org/ -->