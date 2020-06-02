# import numpy as np

from rl_agents.agents.mab import (
    UCB,
    UCB1,
    UCB2,
    DecayEpsilon,
    EpsilonGreedy,
    Pursuit,
    Softmax,
)
from rl_agents.envs import BanditKArmedGaussianEnv
from rl_agents.runners import simple_mab_runner


def test_all_mabs():
    env = BanditKArmedGaussianEnv()
    agents = [
        EpsilonGreedy(env.action_space.n, 0.1),
        DecayEpsilon(env.action_space.n, 0.5, 0.99),
        UCB(env.action_space.n, 0.005),
        UCB1(env.action_space.n),
        UCB2(env.action_space.n, 0.01),
        Softmax(env.action_space.n, 0.02),
        Pursuit(env.action_space.n, 0.1),
    ]
    for agent in agents:
        n_trials = 30
        rewards, regrets, optimals = simple_mab_runner(env, agent, n_trials)
        assert len(rewards) == n_trials
        assert len(regrets) == n_trials
        assert len(optimals) == n_trials
