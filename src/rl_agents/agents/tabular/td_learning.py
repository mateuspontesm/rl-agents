import numpy as np

from rl_agents.agents.core import BaseAgent
from rl_agents.agents.functions import QMatrixFunction
from rl_agents.agents.policies import EGreedyPolicy


class TDAgent(BaseAgent):
    """A base Temporal-Difference Agent.

    This agent is used to build the basic TD algorithms:
        * Q-Learning
        * SARSA
        * Expected-SARSA

    Parameters
    ----------
    n_states : int
        Number of states in the state space.
    n_actions : int
        Number of actions in the action space.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    policy : rl_agents.agents.policies.base.BasePolicy
        A policy object.
    q_function : rl_agents.agents.functions.base.BaseQFunction
        A Q-Function class.

    Attributes
    ----------
    n_states : int
        Number of states.
    n_actions : int
        Number of actions.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    policy : BasePolicy
        Policy instance
    q_function : BaseQFunction
        Qfunction instance

    """

    def __init__(
        self,
        n_states,
        n_actions,
        alpha,
        gamma,
        policy=EGreedyPolicy(0.1),
        q_function=QMatrixFunction,
        q_func_kwargs=None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.policy = policy
        q_func_kwargs = {} if q_func_kwargs is None else q_func_kwargs.copy()
        self.q_function = q_function(self.n_states, self.n_actions, **q_func_kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def predict(self, state, eval=False):
        """Predict the next action the agent should take.

        Parameters
        ----------
        state : int
            Index of the state.
        eval : bool
            Flag to indicate if the agent is in a test setting (evaluation)

        Returns
        -------
        int
            Action index to be taken.

        """
        # Get the Q-Values associated with that state:
        q_values = self.q_function.get_values(state)
        # Utilize the policy:
        if eval:
            action = np.argmax(q_values)
        else:
            action = self.policy(q_values)
        return action

    def learn(self, state, action, reward, next_state):
        r"""Learn from the interaction.

        TD update:

        .. math::
            Q(s,a) \leftarrow (1-\alpha) Q(s,a) + \alpha(r + \gamma F(s))

        The difference between the TD algorithms
        (Q-Learning, SARSA, Expected-SARSA) comes from the difference
        in the :math:`F(s)` function.

        Parameters
        ----------
        state : type
            State in which the action was taken.
        action : type
            Action taken
        reward : float
            Reward received by the transition.
        next_state : type
            Next state the environment transitions.


        """
        # TD update Q(s,a) <- (1-alpha)Q(s,a) + alpha(r + gamma F(s))
        update = self.alpha * (reward + self.gamma * self._next_value(next_state))
        target_q = (1 - self.alpha) * self.q_function(state, action) + update
        # Update the Q-Function with the target:
        self.q_function.update(state, action, target_q)

    def _next_value(self, next_state):
        raise NotImplementedError


class QLearningAgent(TDAgent):
    r"""A Simple Q-Learning Agent

    Refer to `TDAgent` for reference on the parameters and methods.

    The Q-Learning update is given as:

    .. math::
        Q(s, a) \leftarrow  (1-\alpha) Q(s, a) +
                            \alpha \left[ r +
                            \gamma \max_{a' \in A} Q(s', a') \right]

    """

    def __init__(
        self,
        n_states,
        n_actions,
        alpha,
        gamma,
        policy=EGreedyPolicy(0.1),
        q_function=QMatrixFunction,
        q_func_kwargs=None,
    ):

        super().__init__(
            n_states, n_actions, alpha, gamma, policy, q_function, q_func_kwargs,
        )

    def _next_value(self, next_state):
        q_values = self.q_function.get_values(next_state)
        return q_values.max()


class SarsaAgent(TDAgent):
    r"""A Simple SARSA Agent

    Refer to `TDAgent` for reference on the parameters and methods.

    The SARSA update is given as:

    .. math::
        Q(s_t, a_t) \leftarrow  (1-\alpha) Q(s, a) +
        \alpha \left[ r +
        \gamma Q(s',\pi(s') ) \right]

    Where :math:`\pi(s)` returns the action chosen by the policy.

    """

    def __init__(
        self,
        n_states,
        n_actions,
        alpha,
        gamma,
        policy=EGreedyPolicy(0.1),
        q_function=QMatrixFunction,
        q_func_kwargs=None,
    ):

        super().__init__(
            n_states, n_actions, alpha, gamma, policy, q_function, q_func_kwargs,
        )
        self.next_action = None

    def predict(self, state, eval=False):
        """Predict the next action the SARSA agent should take.

        Parameters
        ----------
        state : int
            Index of the state.
        eval : bool
            Flag to indicate if the agent is in a test setting (evaluation)

        Returns
        -------
        int
            Action index to be taken.

        """
        # Get the Q-Values associated with that state:
        q_values = self.q_function.get_values(state)
        # Utilize the policy:
        if eval:
            action = np.argmax(q_values)
        else:
            if self.next_action == None:
                action = self.policy(q_values)
            else:
                action = self.next_action
        return action

    def _next_value(self, next_state):
        q_values = self.q_function.get_values(next_state)
        action = self.policy(q_values)
        self.next_action = action
        return self.q_function(next_state, action)


class ExpectedSarsaAgent(TDAgent):
    r"""A Simple expeted-SARSA Agent

    The Expeted-SARSA update is given as:

    .. math::
        Q(s_t, a_t) \leftarrow  (1-\alpha) Q(s, a) +
        \alpha \left[ r +
        \gamma \sum_{a' \in A} \pi(s',a') Q(s',a') \right]

    Where :math:`\pi(s,a)` returns the probability of taking
    action :math:`a` in state :math:`s`.
    """

    def __init__(
        self,
        n_states,
        n_actions,
        alpha,
        gamma,
        policy=EGreedyPolicy(0.1),
        q_function=QMatrixFunction,
        q_func_kwargs=None,
    ):

        super().__init__(
            n_states, n_actions, alpha, gamma, policy, q_function, q_func_kwargs,
        )

    def _next_value(self, next_state):
        q_values = self.q_function.get_values(next_state)
        pi_values = self.policy.get_values(q_values)
        return np.sum(q_values * pi_values)
