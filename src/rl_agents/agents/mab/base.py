from abc import ABC, abstractmethod


class BaseMAB(ABC):
    """
    A basic Multi-Armed Bandit agent.

    In reinforcement learning, an Agent learns
    by interacting with an Environment.
    Usually, an agent tries to maximize a reward signal.
    It does this by taking "actions" and receiving "rewards",
    and in doing so, learning which action pairs correlate with high rewards.

    An MAB implementation should encapsulate some particular
    reinforcement learning algorihthm.
    """
    @abstractmethod
    def predict(self):
        """Select an action for the current timestep.

        This method allows the agent to do whatever is necessary to select
        an action in a given timestep.
        However, the agent must ultimately return an action (the arm index).

        Returns
        -------
        int
            Index of chosen action.

        """
    @abstractmethod
    def learn(self, a_idx, reward):
        """Learn from the interaction.

        Parameters
        ----------
        a_idx : int
            Index of the arm pulled (action taken).
        reward : float
            Reward received from the system after taking action a_idx.

        """
