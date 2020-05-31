from abc import ABC, abstractmethod

import numpy as np


class BasePolicy(ABC):
    """
    A basic policy for tabular agents.

    The policy is a function that maps a state to an action.

    """

    @abstractmethod
    def __call__(self, q_values):
        """Select an action for the current timestep.

        This method returns the action selected by the current policy

        Returns
        -------
        int
            Index of chosen action.

        """

    @abstractmethod
    def update(self):
        """Updates the policy.

        """


class EGreedyPolicy(BasePolicy):
    """Epsilon-greedy policy for tabular agents.

    Parameters
    ----------
    epsilon : float
        Epsilon parameter to control the exploration-exploitation trade-off.

    Attributes
    ----------
    epsilon

    """

    def __init__(self, epsilon):
        """Instantiate a `EGreedyPolicy` object.

        Parameters
        ----------
        epsilon : float
            Epsilon parameter to control the exploration-exploitation trade-off.


        """
        if (epsilon >= 1) or (epsilon <= 0):
            raise ValueError("Invalid value for epsilon, 0 <= epsilon <= 1")
        self.epsilon = epsilon

    def __call__(self, q_values):
        r"""Select an action based on the :math:`\epsilon`-greedy policy.


        Parameters
        ----------
        q_values : numpy.ndarray(float, ndim=1)
            Line from the Q-Table, corresponding to Q-values
            for the chosen state.

        Returns
        -------
        int
            Chosen action index

        """
        if np.random.rand() < self.epsilon:
            a_idx = np.random.randint(low=0, high=q_values.size)
        else:
            a_idx = np.argmax(q_values)
        return a_idx

    def update(self):
        pass


class EDecreasePolicy(BasePolicy):
    """Decreasing epsilon-greedy policy for tabular agents.

    Parameters
    ----------
    epsilon : float
        Epsilon parameter to control the exploration-exploitation trade-off.

    Attributes
    ----------
    epsilon

    """

    def __init__(self, epsilon, epsilon_min, decay):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay

    def __call__(self, q_values):
        r"""Select an action based on the :math:`\epsilon`-greedy policy.


        Parameters
        ----------
        q_values : numpy.ndarray(float, ndim=1)
            Line from the Q-Table, corresponding to Q-values
            for the chosen state.

        Returns
        -------
        int
            Chosen action index

        """
        if np.random.rand() < self.epsilon:
            a_idx = np.random.randint(low=0, high=q_values.size)
        else:
            a_idx = np.argmax(q_values)
        return a_idx

    def update(self):
        if self.epsilon > self.epsilon_min:
            if self.epsilon * self.decay > self.epsilon_min:
                self.epsilon = self.epsilon * self.decay
            else:
                self.epsilon = self.epsilon_min


class BoltzmanPolicy(BasePolicy):
    r"""Boltzman(Softmax) policy for tabular agents.

    .. math::
        \begin{equation}
        \pi\left(s, a\right) = \frac{e^{Q\left(s, a\right) / T}}
                               {\sum_{i=1}^{m} e^{Q\left(s, a_i\right)/ T}}
        \end{equation}

    Attributes
    ----------
    temperature : float
        Temperature parameter to control the exploration-exploitation trade-off.

    """

    def __init__(self, temperature):
        """Instantiate a `BoltzmanPolicy` object.

        Parameters
        ----------
        temperature : float
            Temperature parameter to control the
            exploration-exploitation trade-off.


        """
        if temperature >= 0:
            raise ValueError("Invalid temperature value, T >= 0")
        self.temperature = temperature

    def __call__(self, q_values):
        r"""Select an action based on the Boltzman policy.

        .. math::
            \begin{equation}
            \pi\left(s, a\right) = \frac{e^{Q\left(s, a\right) / T}}
                                   {\sum_{i=1}^{m} e^{Q\left(s, a_i\right)/ T}}
            \end{equation}

        Parameters
        ----------
        q_values : numpy.ndarray(float, ndim=1)
            Line from the Q-Table, corresponding to Q-values
            for the chosen state.

        Returns
        -------
        int
            Chosen action index

        """
        e_x = np.exp(q_values / self.temperature)
        p_arms = e_x / e_x.sum()
        return np.random.choice(range(q_values.size), p=p_arms)

    def update(self):
        pass
