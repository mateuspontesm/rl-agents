from collections import Counter

import gym
import numpy as np
import pytest

from rl_agents.agents import ExpectedSarsaAgent, QLearningAgent, SarsaAgent
from rl_agents.agents.functions import QMatrixFunction, QTableFunction
from rl_agents.agents.policies import (
    BoltzmanPolicy,
    EDecreasePolicy,
    EGreedyPolicy,
)
from rl_agents.runners import simple_tab_runner


def gen_list():
    pis = [
        EGreedyPolicy(0.1),
        BoltzmanPolicy(10),
        EDecreasePolicy(0.9, 0.1, 0.9),
    ]
    funcs = [QMatrixFunction, QTableFunction]
    ags = [QLearningAgent, SarsaAgent, ExpectedSarsaAgent]
    mets = ["zeros", "ones", "random"]
    return [
        (v1, v2, v3, v4)
        for v1 in ags
        for v2 in pis
        for v3 in funcs
        for v4 in mets
    ]


@pytest.mark.parametrize("AgentC, policy, FunctionC, method", gen_list())
def test_frozen_lake(AgentC, policy, FunctionC, method):
    env = gym.make("FrozenLake-v0")
    q_func_kwargs = {"method": method}
    agent = AgentC(
        n_states=env.nS,
        n_actions=env.nA,
        alpha=0.4,
        gamma=0.5,
        policy=policy,
        q_function=FunctionC,
        q_func_kwargs=q_func_kwargs,
    )
    _ = simple_tab_runner(env, agent, 200)


@pytest.mark.parametrize("epsilon", [0.1, 0.5, 0.9])
def test_egreedy_policy(epsilon):
    n_tests = 10000
    # Best Q value is the index 1
    q_values = np.array([1, 5, 2, 1, 3])
    # Test valid values:
    policy = EGreedyPolicy(epsilon)
    counter = Counter([policy(q_values) for _ in range(n_tests)])
    # Test max value:
    assert counter.most_common(1)[0][0] == 1
    # Test Counting is approx epsilon:
    counts_max = counter.most_common()[0][1]
    counts_min = counter.most_common()[-1][1]
    assert counts_max > (1 - epsilon) * n_tests
    assert counts_min == pytest.approx(epsilon * n_tests / 5, rel=100)
    # Test get_values method:
    assert type(policy.get_values(q_values)) == type(q_values)
    assert len(policy.get_values(q_values)) == len(q_values)


@pytest.mark.parametrize(
    "epsilon, epsilon_min, decay", [(0.9, 0.5, 0.9), (0.5, 0.1, 0.8)]
)
def test_edecrease_policy(epsilon, epsilon_min, decay):
    n_tests = 10000
    # Best Q value is the index 1
    q_values = np.array([1, 5, 2, 1, 3])
    # Test valid values:
    policy = EDecreasePolicy(epsilon, epsilon_min, decay)
    ls = []
    for _ in range(n_tests):
        ls.append(policy(q_values))
        policy.update()
    counter = Counter(ls)
    # Test max value:
    assert counter.most_common(1)[0][0] == 1
    assert policy.epsilon == epsilon_min


@pytest.mark.parametrize("temperature", [10, 5, 1, 0.8])
def test_boltzman_policy(temperature):
    n_tests = 10000
    # Best Q value is the index 1
    q_values = np.array([1, 5, 2, 1.5, 3])
    true_order = [1, 4, 2, 3, 0]
    # Test valid values:
    policy = BoltzmanPolicy(temperature)
    counter = Counter([policy(q_values) for _ in range(n_tests)])
    print(counter.most_common())
    print(policy.get_values(q_values))
    # Test max value:
    assert counter.most_common(1)[0][0] == 1
    # Test Order:
    counter_order = [n[0] for n in counter.most_common()]
    assert counter_order == true_order
    # Test get_values method:
    assert type(policy.get_values(q_values)) == type(q_values)
    assert len(policy.get_values(q_values)) == len(q_values)


def test_policies_errors():
    with pytest.raises(ValueError):
        EGreedyPolicy(-0.1)
    with pytest.raises(ValueError):
        EGreedyPolicy(1.1)
    with pytest.raises(ValueError):
        EDecreasePolicy(1.1, 0.1, 0.9)
    with pytest.raises(ValueError):
        EDecreasePolicy(-0.2, 0.2, 0.8)
    with pytest.raises(ValueError):
        BoltzmanPolicy(-500)


def test_functions_errors():
    with pytest.raises(ValueError):
        QMatrixFunction(2, 2, "blabla")
    with pytest.raises(ValueError):
        QTableFunction(2, 2, "blablo")
