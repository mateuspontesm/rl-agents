"""
The :mod:`rl_agents.agents.functions` submodule includes:

* A basic tabular Q-function
* A base class for implementing different functions
"""
from rl_agents.agents.functions.tabular_functions import (
    QMatrixFunction,
    QTableFunction,
)

__all__ = ["QMatrixFunction", "QTableFunction"]
