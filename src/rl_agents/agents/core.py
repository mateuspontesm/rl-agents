from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """A base RL agent.

    """

    @abstractmethod
    def predict(self, state):
        """Predict the next action the agent should take.

        Parameters
        ----------
        state : type
            State information

        Returns
        -------
        type
            Action to be taken.

        """

    @abstractmethod
    def learn(self, state, action, reward, next_state):
        """Learn from the interaction.

        Parameters
        ----------
        state : type
            State information.
        action : type
            Action taken in the current step.
        reward : type
            Reward received for the action taken.
        next_state : type
            Next state of the environment.


        """
