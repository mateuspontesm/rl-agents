import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class BanditEnv(gym.Env):
    """
    Bandit environment base

    Attributes
    ----------
    arms: int
        Number of arms
    """

    def __init__(self, arms: int):
        self.arms = arms
        self.action_space = spaces.Discrete(self.arms)
        self.observation_space = spaces.Discrete(1)
        self.np_random = None
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        raise NotImplementedError

    def step(self, action: int):
        raise NotImplementedError

    def render(self, mode="human", close=False):
        raise NotImplementedError


class BanditKArmedGaussianEnv(BanditEnv):
    """
    K-armed bandit environment.

    This env is mentioned on page 30 of Sutton and Barto's with K=10
    [Reinforcement Learning: An Introduction]
    (https://www.dropbox.com/s/b3psxv2r0ccmf80/book2015oct.pdf?dl=0)

    Actions always pay out
    Mean of payout is pulled from a normal distribution (0, 1) (called q*(a))
    Actual reward is drawn from a normal distribution (q*(a), 1)
    """

    def __init__(self, arms=10):
        self.means = []
        BanditEnv.__init__(self, arms)

    def reset(self):
        self.means = []
        for _ in range(self.arms):
            self.means.append(self.np_random.normal(0, 1))
        self.best_arm = np.argmax(self.means)

    def step(self, action: int):
        assert self.action_space.contains(action)
        reward = self.np_random.normal(self.means[action], 1)
        info = {
            "Optimal": action == self.best_arm,
            "Regret": self.means[self.best_arm] - self.means[action],
        }
        return 0, reward, False, info
