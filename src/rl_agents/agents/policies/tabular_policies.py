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

        Parameters
        ----------
        q_values : numpy.ndarray(float, ndim=1)
            Q-value of each action.

        Returns
        -------
        int
            Index of chosen action.

        """

    @abstractmethod
    def update(self):
        """Update the policy.

        """

    @abstractmethod
    def get_values(self, q_values):
        """Return the probabilities associated with each action.

        Parameters
        ----------
        q_values : numpy.ndarray(float, ndim=1)
            Q-value of each action.

        Returns
        -------
        numpy.ndarray(float, ndim=1)
            Probabilities associated with each action.

        """


class EGreedyPolicy(BasePolicy):
    """Epsilon-greedy policy for tabular agents.

    Parameters
    ----------
    epsilon : float
        Epsilon parameter to control the exploration-exploitation trade-off.

    Attributes
    ----------
    epsilon : float
        Exploration-exploitation parameter.

    """

    def __init__(self, epsilon):
        """Instantiate a `EGreedyPolicy` object.

        Parameters
        ----------
        epsilon : float
            Epsilon parameter to control the
            exploration-exploitation trade-off.


        """
        # Check for valid values:
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
        # Explorarion case:
        if np.random.rand() < self.epsilon:
            a_idx = np.random.randint(low=0, high=q_values.size)
        # Exploitation case:
        else:
            a_idx = np.argmax(q_values)
        return a_idx

    def update(self):
        pass

    def get_values(self, q_values):
        # Probababilites of exploration:
        output = np.ones(q_values.size) * self.epsilon / q_values.size
        # Add the exploitation probability:
        output[q_values.argmax()] += 1 - self.epsilon
        return output


class EDecreasePolicy(EGreedyPolicy):
    """Decreasing epsilon-greedy policy for tabular agents.

    Parameters
    ----------
    epsilon : float
        Epsilon parameter to control the exploration-exploitation trade-off.
    epsilon_min : float
        Minimum epsilon acceptable
    decay : float
        Decay for epsilon: epsilon <- epsilon * decay

    Attributes
    ----------
    epsilon_min : float
    decay : float

    """

    def __init__(self, epsilon, epsilon_min, decay):
        super().__init__(epsilon)
        self.epsilon_min = epsilon_min
        self.decay = decay

    # The call method is the same as the epsilon-greedy

    def update(self):
        if self.epsilon > self.epsilon_min:
            if self.epsilon * self.decay > self.epsilon_min:
                self.epsilon = self.epsilon * self.decay
            else:
                self.epsilon = self.epsilon_min

    # The get_values method is the same as the epsilon-greedy


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
        Temperature parameter to control the
        exploration-exploitation trade-off.

    """

    def __init__(self, temperature):
        """Instantiate a `BoltzmanPolicy` object.

        Parameters
        ----------
        temperature : float
            Temperature parameter to control the
            exploration-exploitation trade-off.


        """
        # Check for valid values:
        if temperature < 0:
            raise ValueError("Invalid temperature value, T >= 0")
        self.temperature = temperature

    def __call__(self, q_values):
        r"""Select an action based on the Boltzman policy.

        .. math::
            \pi\left(s, a\right) = \frac{e^{Q\left(s, a\right) / T}}
                                   {\sum_{i=1}^{m} e^{Q\left(s, a_i\right)/ T}}

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
        # Exponential:
        e_x = np.exp(q_values / self.temperature)
        # Probabilities
        p_arms = e_x / e_x.sum()
        return np.random.choice(range(q_values.size), p=p_arms)

    def update(self):
        pass

    def get_values(self, q_values):
        e_x = np.exp(q_values / self.temperature)
        p_arms = e_x / e_x.sum()
        return p_arms
