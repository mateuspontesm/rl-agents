from collections import defaultdict

import numpy as np

from rl_agents.agents.functions.base import BaseQFunction


class QMatrixFunction(BaseQFunction):
    """A simple Q-table using numpy array.

    Parameters
    ----------
    n_states : int
        Number of states in the state space.
    n_actions : int
        Number of actions in the action space.
    method : str
        Initialization method. Possible values: 'zeros', 'random' or 'ones'.

    Attributes
    ----------
    q_table : numpy.ndarray(float, ndims=2)
        Q-Table matrix, rows are the states and columns the actions.

    """

    def __init__(self, n_states, n_actions, method="zeros"):
        # if method not in ["zeros", "ones", "random"]:

        if method == "zeros":
            self.q_table = np.zeros((n_states, n_actions))
        elif method == "random":
            self.q_table = np.random.random((n_states, n_actions))
        elif method == "ones":
            self.q_table = np.ones((n_states, n_actions))
        else:
            raise ValueError(
                "Invalid Method, options: 'zeros', 'random' or 'ones'. "
            )

    def __call__(self, state, action):
        return self.q_table[state, action]

    def update(self, state, action, target):
        self.q_table[state, action] = target

    def get_values(self, state):
        return self.q_table[state, :]


class QTableFunction(BaseQFunction):
    """A simple Q-table using `defaultdict`.

    Parameters
    ----------
    n_states : int
        Number of states in the state space.
    n_actions : int
        Number of actions in the action space.
    method : str
        Initialization method. Possible values: 'zeros', 'random' or 'ones'.

    Attributes
    ----------
    q_table : defaultdict
        Each key is a vector representing the Q-values
        of each action for that state.

    """

    def __init__(self, n_states, n_actions, method="zeros"):
        if method == "zeros":
            self.q_table = defaultdict(lambda: np.zeros(n_actions))
        elif method == "random":
            self.q_table = defaultdict(lambda: np.random.random(n_actions))
        elif method == "ones":
            self.q_table = defaultdict(lambda: np.ones(n_actions))
        else:
            raise ValueError(
                "Invalid Method, options: 'zeros', 'random' or 'ones'. "
            )

    def __call__(self, state, action):
        return self.q_table[state][action]

    def update(self, state, action, target):
        self.q_table[state][action] = target

    def get_values(self, state):
        return self.q_table[state]
