import numpy as np

from rl_agents.agents.mab.base import BaseMAB


class Softmax(BaseMAB):
    """Short summary.

    Parameters
    ----------
    n_arms : type
        Description of parameter `n_arms`.
    temperature : type
        Description of parameter `temperature`.

    Attributes
    ----------
    means : type
        Description of attribute `means`.
    p_arms : type
        Description of attribute `p_arms`.
    trials : type
        Description of attribute `trials`.
    n_arms
    temperature

    """

    def __init__(self, n_arms, temperature):
        self.n_arms = n_arms
        self.means = np.zeros(self.n_arms)
        self.temperature = temperature
        self.p_arms = np.zeros(self.n_arms)
        self.trials = np.zeros(self.n_arms)

    def learn(self, a_idx, reward):
        """Short summary.

        Parameters
        ----------
        a_idx : type
            Description of parameter `a_idx`.
        reward : type
            Description of parameter `reward`.

        Returns
        -------
        type
            Description of returned object.

        """
        self.means[a_idx] = (
            (self.means[a_idx] * self.trials[a_idx]) + reward
        ) / (self.trials[a_idx] + 1)
        self.trials[a_idx] += 1  # add trial

    def predict(self):
        """Short summary.

        Returns
        -------
        type
            Description of returned object.

        """
        e_x = np.exp(self.means / self.temperature)
        self.p_arms = e_x / e_x.sum()
        return np.random.choice(range(self.n_arms), p=self.p_arms)
