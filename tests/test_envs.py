# import numpy as np

from rl_agents.envs import BanditKArmedGaussianEnv


def test_k_armed_env():
    env = BanditKArmedGaussianEnv()
    _ = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
