"""
The :mod:`rl_agents.agents.policies` submodule includes:

* Epsilon Greedy
* Boltzman (a.k.a Softmax)
"""
from rl_agents.agents.policies.tabular_policies import (
    BoltzmanPolicy,
    EDecreasePolicy,
    EGreedyPolicy,
)

__all__ = ["EGreedyPolicy", "EDecreasePolicy", "BoltzmanPolicy"]
