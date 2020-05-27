"""
The :mod:`rl_agents.agents.mab` submodule includes:

* Epsilon Greedy
* Decreasing Epsilon Greedy
* UCBs: UCB, UCB1, UCB2
* Softmax
* Pursuit
"""


from rl_agents.agents.mab.egreedy import DecayEpsilon  # noqa: F401
from rl_agents.agents.mab.egreedy import EpsilonGreedy  # noqa: F401
from rl_agents.agents.mab.pursuit import Pursuit  # noqa: F401
from rl_agents.agents.mab.softmax import Softmax  # noqa: F401
from rl_agents.agents.mab.ucbs import UCB, UCB1, UCB2  # noqa: F401
