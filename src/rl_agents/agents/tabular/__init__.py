"""
The :mod:`rl_agents.agents.tabular` submodule includes:

* Q-Learning
* SARSA
* Expected SARSA
"""
from rl_agents.agents.tabular.td_learning import (  # isort:skip
    QLearningAgent,
    SarsaAgent,
    ExpectedSarsaAgent,
)

__all__: ["QLearningAgent", "SarsaAgent", "ExpectedSarsaAgent"]
