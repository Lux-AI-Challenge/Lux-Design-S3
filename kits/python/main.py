import json
from typing import Dict
import sys
from argparse import Namespace

import numpy as np

from agent import Agent
from lux.config import EnvConfig
from lux.kit import GameState, process_obs, to_json, from_json, process_action, obs_to_game_state
### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = dict() # store potentially multiple dictionaries as kaggle imports code directly
agent_prev_obs = dict()
def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    step = observation.step
    
    
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        env_cfg = EnvConfig.from_dict(configurations["env_cfg"])
        agent_dict[player] = Agent(player, env_cfg)
        agent_prev_obs[player] = dict()
        agent = agent_dict[player]
    agent = agent_dict[player]
    obs = process_obs(player, agent_prev_obs[player], step, json.loads(observation.obs))
    agent_prev_obs[player] = obs
    agent.step = step
    if obs["real_env_steps"] < 0:
        actions = agent.early_setup(step, obs, remainingOverageTime)
    else:
        actions = agent.act(step, obs, remainingOverageTime)

    return process_action(actions)

if __name__ == "__main__":
    
    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)
    step = 0
    player_id = 0
    env_cfg = None
    i = 0
    # with open("inputs.txt", "w") as f:
    #     f.write("test")
    while True:
        inputs = read_input()
        # with open(f"inputs_{i}.txt", "w") as f:
        #     f.write(inputs)
        
        # print(inputs)
        # observation = Namespace(**dict(step=obs["step"], obs=json.dumps(obs["obs"]), remainingOverageTime=obs["remainingOverageTime"], player=obs["player"], info=obs["info"]))
        if i == 0:
            obs = json.loads(inputs)
            env_cfg = obs["info"]["env_cfg"]
        i += 1
        actions = np.zeros(env_cfg["max_units"], dtype=int) + 1
        # actions = agent_fn(observation, dict(env_cfg=configurations))
        # # send actions to engine
        print(json.dumps(dict(action=actions.tolist())))
        # print([0, 0, 0, 0, 0, 0])
