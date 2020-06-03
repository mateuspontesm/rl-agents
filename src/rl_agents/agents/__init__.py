"""
The :mod:`rl_agents.agents` module includes the RL agents
classes and utilities. It includes the MAB variants,
tabular methods and Deep RL.
"""

import rl_agents.agents.mab

from rl_agents.agents.tabular import (  # isort:skip
    QLearningAgent,
    SarsaAgent,
    ExpectedSarsaAgent,
)

__all__ = ["QLearningAgent", "SarsaAgent", "ExpectedSarsaAgent"]
