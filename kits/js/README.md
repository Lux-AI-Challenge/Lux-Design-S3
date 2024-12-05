# Lux AI Season 3 JS Kit

This is the folder for the JS kit. Please make sure to read the instructions as they are important regarding how you will write a bot and submit it to the competition.

Make sure to check our [Discord](https://discord.gg/aWJt3UAcgn) for announcements if there are any breaking changes.
 or the [Kaggle forums](https://www.kaggle.com/c/lux-ai-season-3/discussion) 
## Requirements

You will need Node 18 or higher installed. You can install Node using [nvm](https://github.com/nvm-sh/nvm).

## Getting Started

To get started, download this folder from this repository.

Your core agent code will go into `agent.js`, and you can create and use more files to help you as well. You should leave `main.py` alone as that code enables your agent to compete against other agents locally and on Kaggle.

To quickly test run your agent, run

```
luxai-s3 main.py main.py --output=replay.json
```

This will run the `agent.js` code in the same folder as `main.py` and generate a replay file saved to `replay.json`.

## Developing
Now that you have the code up and running, you are ready to start programming and having some fun!

If you haven't read it already, take a look at the [design specifications for the competition](../../docs/specs.md). This will go through the rules and objectives of the competition. 

All of our kits follow a common API through which you can use to access various functions and properties that will help you develop your strategy and bot. The markdown version is here: https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/kits/README.md, which also describes the observation and action structure/spaces.

## Submitting to Kaggle

Submissions need to be a .tar.gz bundle with main.py at the top level directory (not nested). To create a submission, create the .tar.gz with `tar -czvf submission.tar.gz *`. Upload this under the [My Submissions tab](https://www.kaggle.com/competitions/lux-ai-season-3/submissions) and you should be good to go! Your submission will start with a scheduled game vs itself to ensure everything is working before being entered into the matchmaking pool against the rest of the leaderboard.