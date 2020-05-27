import numpy as np

from rl_agents.agents.mab.base import BaseMAB


class Pursuit(BaseMAB):
    """Short summary.

    Parameters
    ----------
    n_arms : type
        Description of parameter `n_arms`.
    beta : type
        Description of parameter `beta`.

    Attributes
    ----------
    means : type
        Description of attribute `means`.
    p_arms : type
        Description of attribute `p_arms`.
    trials : type
        Description of attribute `trials`.
    n_arms
    beta

    """

    def __init__(self, n_arms, beta):
        self.n_arms = n_arms
        self.means = np.zeros(self.n_arms)
        self.beta = beta
        self.p_arms = np.ones(self.n_arms) / self.n_arms
        self.trials = np.zeros(self.n_arms)

    def learn(self, a_idx, reward):
        """Short summary.

        Parameters
        ----------
        reward : type
            Description of parameter `reward`.
        a_idx : type
            Description of parameter `a_idx`.

        Returns
        -------
        type
            Description of returned object.

        """
        self.means[a_idx] = (
            (self.means[a_idx] * self.trials[a_idx]) + reward
        ) / (self.trials[a_idx] + 1)
        self.trials[a_idx] += 1  # add trial
        ii = np.argmax(self.means)
        p_ii = self.p_arms[ii]
        self.p_arms = (1 - self.beta) * self.p_arms
        self.p_arms[ii] = p_ii + self.beta * (1 - p_ii)

    def predict(self):
        """Short summary.

        Returns
        -------
        type
            Description of returned object.

        """
        return np.random.choice(range(self.n_arms), p=self.p_arms)
