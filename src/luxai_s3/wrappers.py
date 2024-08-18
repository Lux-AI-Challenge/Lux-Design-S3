# TODO (stao): Add lux ai s3 env to gymnax api wrapper, which is the old gym api
import gymnasium as gym
import gymnax


class RecordEpisode(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_returns = []
        self.episode_lengths = []