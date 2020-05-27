import numpy as np

from rl_agents.agents.mab.base import BaseMAB


class EpsilonGreedy(BaseMAB):
    r"""Epsilon-Greedy agent.

    The agent uses the epsilon-greedy approach to solve the Multi-Armed
    bandit problem.

    The parameter :math:`\epsilon` is used for the
    exploration-exploitation trade-off. With probability :math:`\epsilon`
    the agent selects a random action, otherwise it selects the action
    that has the best average reward.

    Parameters
    ----------
    n_arms : int
        Number of actions (arms) of the MAB.
    epsilon : float
        Probability of selecting a random action.

    Attributes
    ----------
    means : numpy.array(float, ndim=1)
        Vector containing the average reward of each arm.
    trials : numpy.array(float, ndim=1)
        Vector containing the number of trials made to each arm.

    """

    def __init__(self, n_arms, epsilon):
        self.epsilon = epsilon
        self.n_arms = n_arms
        self.means = np.zeros(self.n_arms)
        self.trials = np.zeros(self.n_arms)

    def learn(self, a_idx, reward):
        """Make `EpsilonGreedy` agent learn from the interaction.

        The `EpsilonGreedy` agent learns from its previous choice
        and the reward received from this action.
        Updates the means and the trials.

        Parameters
        ----------
        reward : float
            Reward received from the system after taking action a_idx.
        a_idx : int
            Index of the arm pulled (action taken).

        """
        self.means[a_idx] = (
            self.means[a_idx] * self.trials[a_idx] + reward
        ) / (self.trials[a_idx] + 1)
        self.trials[a_idx] += 1  # add trial

    def predict(self):
        r"""Predict next action.

        With probability :math:`\epsilon` the agent selects a random arm.
        With probability :math:`1 - \epsilon` the agent selects the arm that
        has the best average reward.

        Returns
        -------
        int
            Index of chosen action.

        """
        if np.random.rand() < self.epsilon:
            a = np.random.randint(low=0, high=self.n_arms)
        else:
            a = self.means.argmax()
        return a


class DecayEpsilon(BaseMAB):
    r"""Agent that follows an epsilon-decreasing policy.

    The agent uses the epsilon-greedy approach to solve the Multi-Armed
    bandit problem, but with a decay in the epsilon.

    The parameter :math:`\epsilon` is used for the exploration-exploitation
    trade-off. With probability :math:`\epsilon` the agent selects a
    random action, otherwise it selects the action that has the
    best average reward. After each interaction the epsilon is updated as
    epsilon = epsilon * decay.

    Parameters
    ----------
    n_arms : int
        Number of actions (arms) of the MAB.
    max_epsilon : float
        Initial epsilon.
    decay : float
        Decay of the epsilon.

    Attributes
    ----------
    epsilon : float
        Epsilon of the agent. Constantly updated as epsilon = epsilon*decay
    means : numpy.array(float, ndim=1)
        Vector containing the average reward of each arm.
    trials : numpy.array(float, ndim=1)
        Vector containing the number of trials made to each arm.

    """

    def __init__(self, n_arms, max_epsilon, decay):
        self.epsilon = max_epsilon
        self.n_arms = n_arms
        self.means = np.zeros(self.n_arms)
        self.trials = np.zeros(self.n_arms)
        self.decay = decay

    def learn(self, a_idx, reward):
        """Make the `DecayEpsilon` agent learn from the interaction.

        The MAB agent learns from its previous choice and the reward received
        from this action. Updates the means and the trials.

        Parameters
        ----------
        reward : float
            Reward received from the system after taking action a_idx.
        a_idx : int
            Index of the arm pulled (action taken).

        """
        self.means[a_idx] = (
            self.means[a_idx] * self.trials[a_idx] + reward
        ) / (self.trials[a_idx] + 1)
        self.trials[a_idx] += 1  # add trial

    def predict(self):
        r"""Predict next action and update epsilon.

        With probability :math:`\epsilon` the agent selects a random arm.
        With probability :math:`1 - \epsilon` the agent selects the arm that
        has the best average reward.

        Returns
        -------
        int
            Index of chosen action.

        """
        if np.random.rand() < self.epsilon:
            a_idx = np.random.randint(low=0, high=self.n_arms)
        else:
            a_idx = self.means.argmax()
        self.epsilon = self.epsilon * self.decay
        return a_idx
