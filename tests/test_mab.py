import numpy as np

from rl_agents.envs import BanditKArmedGaussianEnv
from rl_agents.runners import simple_mab_runner
from rl_agents.agents.mab import (
    EpsilonGreedy,
    DecayEpsilon,
    Softmax,
    Pursuit,
    UCB,
    UCB1,
    UCB2,
)


def test_egreedy():
    env = BanditKArmedGaussianEnv()
    agent = EpsilonGreedy(env.action_space.n, 0.1)
    n_trials = 30
    rewards, regrets, optimals = simple_mab_runner(env, agent, n_trials)
    assert len(rewards) == n_trials
    assert len(regrets) == n_trials
    assert len(optimals) == n_trials


def test_decay_greedy():
    env = BanditKArmedGaussianEnv()
    agent = DecayEpsilon(env.action_space.n, 0.5, 0.99)
    n_trials = 30
    rewards, regrets, optimals = simple_mab_runner(env, agent, n_trials)
    assert len(rewards) == n_trials
    assert len(regrets) == n_trials
    assert len(optimals) == n_trials


def test_ucb():
    env = BanditKArmedGaussianEnv()
    agent = UCB(env.action_space.n, 0.005)
    n_trials = 30
    rewards, regrets, optimals = simple_mab_runner(env, agent, n_trials)
    assert len(rewards) == n_trials
    assert len(regrets) == n_trials
    assert len(optimals) == n_trials


def test_ucb1():
    env = BanditKArmedGaussianEnv()
    agent = UCB1(env.action_space.n)
    n_trials = 30
    rewards, regrets, optimals = simple_mab_runner(env, agent, n_trials)
    assert len(rewards) == n_trials
    assert len(regrets) == n_trials
    assert len(optimals) == n_trials


def test_ucb2():
    env = BanditKArmedGaussianEnv()
    agent = UCB2(env.action_space.n, 0.01)
    n_trials = 30
    rewards, regrets, optimals = simple_mab_runner(env, agent, n_trials)
    assert len(rewards) == n_trials
    assert len(regrets) == n_trials
    assert len(optimals) == n_trials


def test_softmax():
    env = BanditKArmedGaussianEnv()
    agent = Softmax(env.action_space.n, 0.02)
    n_trials = 30
    rewards, regrets, optimals = simple_mab_runner(env, agent, n_trials)
    assert len(rewards) == n_trials
    assert len(regrets) == n_trials
    assert len(optimals) == n_trials


def test_pursuit():
    env = BanditKArmedGaussianEnv()
    agent = Pursuit(env.action_space.n, 0.1)
    n_trials = 30
    rewards, regrets, optimals = simple_mab_runner(env, agent, n_trials)
    assert len(rewards) == n_trials
    assert len(regrets) == n_trials
    assert len(optimals) == n_trials
