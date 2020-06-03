from abc import ABC, abstractmethod


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
