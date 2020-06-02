from abc import ABC, abstractmethod
from collections import default_dict

import numpy as np


class BaseQFunction(ABC):
    """
    A basic Q-function RL agents.

    The Q-Function is function that maps a state-action pair to a real value.

    """

    @abstractmethod
    def __call__(self, state, action):
        """Get the Q-value of the state-action pair.

        Parameters
        ----------
        state : type
            The state information
        action : type
            The action information

        Returns
        -------
        float
            Q-Value of the state-action pair.

        """

    @abstractmethod
    def update(self):
        """Update the Q-function.

        """

    @abstractmethod
    def get_values(self, state):
        """Return the Q-Values of all actions in the selected state.

        Parameters
        ----------
        state : type
            State information

        Returns
        -------
        numpy.ndarray(float, ndim=1)
            Q-Values of the actions associated with the state.

        """


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
        if method not in ["zeros", "ones", "random"]:
            raise ValueError(
                "Invalid Method, options: 'zeros', 'random' or 'ones'. "
            )
        if method == "zeros":
            self.q_table = np.zeros((n_states, n_actions))
        elif method == "random":
            self.q_table = np.random.random((n_states, n_actions))
        elif method == "ones":
            self.q_table = np.ones((n_states, n_actions))

    def __call__(self, state, action):
        return self.q_table[state, action]

    def update(self, state, action, target):
        self.q_table[state, action] = target


class QTableFunction(BaseQFunction):
    """A simple Q-table using `default_dict`.

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
    q_table : default_dict
        Each key is a vector representing the Q-values
        of each action for that state.

    """

    def __init__(self, n_states, n_actions, method="zeros"):
        if method not in ["zeros", "ones", "random"]:
            raise ValueError(
                "Invalid Method, options: 'zeros', 'random' or 'ones'. "
            )
        if method == "zeros":
            self.q_table = default_dict(lambda: np.zeros(n_actions))
        elif method == "random":
            self.q_table = default_dict(lambda: np.random.random(n_actions))
        elif method == "ones":
            self.q_table = default_dict(lambda: np.ones(n_actions))

    def __call__(self, state, action):
        return self.q_table[state][action]

    def update(self, state, action, target):
        self.q_table[state][action] = target
