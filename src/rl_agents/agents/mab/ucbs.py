import numpy as np

from rl_agents.agents.mab.base import BaseMAB


class UCB(BaseMAB):
    r"""MAB Agent following a Upper Confidence Bound policy.

    The UCB selects the action that maximizes the function given by:

    .. math:: f(i) = \mu_i + U_i,

    where  :math:`\mu_i` is the average reward of arm :math:`i`, and
    :math:`U_i` is given by:

    .. math:: U_i = \sqrt{\frac{-\log{p}}{2 N_i} },

    where :math:`N_i` is the number of pulls made to arm  :math:`i`.

    Parameters
    ----------
    n_arms : int
        Number of actions (arms) of the MAB.
    p : float
        Probability of the true value being above the estimate plus the bound.

    Attributes
    ----------
    means : numpy.array(float, ndim=1)
        Vector containing the average reward of each arm.
    trials : numpy.array(float, ndim=1)
        Vector containing the number of trials made to each arm.
    bounds : numpy.array(float, ndim=1)
        Vector containing the upper bounds of each arm.
    t : int
        Total trial counter.

    """

    def __init__(self, n_arms, p):
        self.p = p
        self.n_arms = n_arms
        self.means = np.zeros(self.n_arms)
        self.trials = np.zeros(self.n_arms)
        self.bounds = np.zeros(self.n_arms)
        self.t = 0

    def learn(self, a_idx, reward):
        """Learn from the interaction.

        Update the means, the bounds and the trials.

        Parameters
        ----------
        reward : float
            Reward received from the system after taking action a_idx.
        a_idx : int
            Index of the arm pulled (action taken).

        """
        self.means[a_idx] = (
            (self.means[a_idx] * self.trials[a_idx]) + reward
        ) / (self.trials[a_idx] + 1)
        self.trials[a_idx] += 1  # add trial
        self.bounds[a_idx] = np.sqrt(-np.log(self.p) / 2 * self.trials[a_idx])
        self.t += 1

    def predict(self):
        """Predict next action.

        Pulls each arm once, then chooses the arm that gives the best
        mean + bound.

        Returns
        -------
        int
            Index of chosen action.

        """
        if self.t < self.n_arms:
            return self.t
        return np.argmax(self.means + self.bounds)


class UCB1(BaseMAB):
    """Short summary.

    Parameters
    ----------
    n_arms : type
        Description of parameter `n_arms`.
    c : type
        Description of parameter `c`.

    Attributes
    ----------
    means : type
        Description of attribute `means`.
    trials : type
        Description of attribute `trials`.
    bounds : type
        Description of attribute `bounds`.
    t : type
        Description of attribute `t`.
    n_arms
    c

    """

    def __init__(self, n_arms, c=4):
        self.n_arms = n_arms
        self.means = np.zeros(self.n_arms)
        self.trials = np.zeros(self.n_arms)
        self.bounds = np.zeros(self.n_arms)
        self.c = c
        self.t = 0

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
        self.t += 1
        self.bounds[a_idx] = self.c * np.sqrt(
            np.log(self.t) / (self.trials[a_idx])
        )

    def predict(self):
        """Short summary.

        Returns
        -------
        type
            Description of returned object.

        """
        if self.t < self.n_arms:
            return self.t
        return np.argmax(self.means + self.bounds)


class UCB2(BaseMAB):
    """Short summary.

    Parameters
    ----------
    n_arms : type
        Description of parameter `n_arms`.
    alpha : type
        Description of parameter `alpha`.

    Attributes
    ----------
    means : type
        Description of attribute `means`.
    trials : type
        Description of attribute `trials`.
    bounds : type
        Description of attribute `bounds`.
    rj : type
        Description of attribute `rj`.
    t : type
        Description of attribute `t`.
    counter : type
        Description of attribute `counter`.
    current : type
        Description of attribute `current`.
    n_arms
    alpha

    """

    def __init__(self, n_arms, alpha):
        self.n_arms = n_arms
        self.means = np.zeros(self.n_arms)
        self.trials = np.zeros(self.n_arms)
        self.bounds = np.zeros(self.n_arms)
        self.rj = np.zeros(self.n_arms)
        self.alpha = alpha
        self.t = 0
        self.counter = 0
        self.current = 0

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
        self.t += 1
        tau = self._tau(self.rj[a_idx])
        self.bounds[a_idx] = np.sqrt(
            (1 + self.alpha) * np.log(np.e * self.t / tau) / (2 * tau)
        )
        self.counter = self._tau(self.rj[a_idx] + 1) - tau
        self.rj[a_idx] += 1

    def predict(self):
        """Short summary.

        Returns
        -------
        type
            Description of returned object.

        """
        if self.t < self.n_arms:
            return self.t
        if self.counter == 0:
            action = np.argmax(self.means + self.bounds)
            self.current = action
            return action
        else:
            self.counter -= 1
            return self.current

    def _tau(self, rj):
        return np.ceil((1 + self.alpha) ** rj)
