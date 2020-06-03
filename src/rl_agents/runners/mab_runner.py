import numpy as np


def simple_mab_runner(env, agent, n_trials):
    """Run a simple MAB experiment.

    Parameters
    ----------
    env : gym.Env
        Environment.
    agent : rl_agents.agents.mab.BaseMAB
        MAB agent.
    n_trials : int
        Number of trials until the experiment ends.

    Returns
    -------
    rewards : numpy.ndarray(float, ndims=1)
        Vector with the observed rewards.
    regrets : numpy.ndarray(float, ndims=1)
        Vector with the regret of each trial.
    optimal : numpy.ndarray(float, ndims=1)
        Vector containing the percentage of optimal actions taken.

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
