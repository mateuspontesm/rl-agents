import numpy as np
from tqdm import tqdm


def simple_tab_runner(env, agent, n_episodes):
    """Short summary.

    Parameters
    ----------
    env : gym.Env
        Environment.
    agent : rl_agents.agents.BaseAgent
        MAB agent.
    n_episodes : int
        Number of episodes to run.

    Returns
    -------
    rewards : numpy.ndarray(float, ndims=1)
        Vector with the total episode reward.

    """
    rewards = np.zeros(n_episodes)
    for ii in tqdm(range(n_episodes)):
        # Run episode:
        done = False
        episode_reward = 0
        obs = env.reset()
        while not done:
            action = agent.predict(obs)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            agent.policy.update()
            obs = next_obs
            episode_reward += reward
        rewards[ii] = episode_reward
    return rewards
