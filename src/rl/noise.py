import numpy as np


class OUNoise(object):
    """
    Noise added to help with model exploration, in reinforcement learning to improve generality, noise is added
    to the action states in order to explore a larger state space.

    Ornstein-Ulhenbeck Process
    Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
    More documentation: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """

    def __init__(self, action_space: np.ndarray, mu: float = 0.0, theta: float = 0.15, max_sigma: float = 0.3,
                 min_sigma: float = 0.3, decay_period: int = 100000) -> None:
        """
        Initializes the Ornstein-Ulhenbeck Process noise
        :param action_space: the min and max values that the action can be, column 1 is the action minimums,
        column 0 is the action maximums
        :param mu: mean shift
        :param theta: the update factor to scale dx
        :param max_sigma: the maximum variance
        :param min_sigma: the minimum variance
        :param decay_period: randomness decay factor
        """
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space[:, 0]
        self.high = action_space[:, 1]

        self.state = None
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal noise state
        """
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self) -> np.ndarray:
        """
        Updates the random internal state by a small dx
        :return: The new internal state that was updated
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action: np.ndarray, t: int = 0) -> np.ndarray:
        """
        Modifies the action using the internal random state vector
        :param action:
        :param t: the current step for the decay factor
        :return: the updated action vector with the added noise
        """
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
