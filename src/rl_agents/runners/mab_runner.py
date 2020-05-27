import numpy as np


def simple_mab_runner(env, agent, n_trials):
    """Short summary.

    Parameters
    ----------
    env : type
        Description of parameter `env`.
    agent : type
        Description of parameter `agent`.
    n_trials : type
        Description of parameter `n_trials`.

    Returns
    -------
    type
        Description of returned object.

    """
    _ = env.reset()
    regrets = np.zeros(n_trials)
    rewards = np.zeros(n_trials)
    optimals = np.zeros(n_trials)
    for ii in range(n_trials):
        arm_idx = agent.predict()
        _, reward, _, info = env.step(arm_idx)
        agent.learn(arm_idx, reward)
        rewards[ii] = (rewards.sum() + reward) / (ii + 1)
        regrets[ii] = (regrets.sum() + info["Regret"]) / (ii + 1)
        optimals[ii] = (optimals.sum() + info["Optimal"]) / (ii + 1)
    return rewards, regrets, optimals
