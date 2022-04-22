import random
from collections import deque
from typing import Sequence

import kdtree
import numpy as np


class Memory:
    """
    Basic uniform sampling technique
    """

    def __init__(self, max_size) -> None:
        """
        Initializes memory buffer
        :param max_size: the maximum buffer size for the memory
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state: Sequence[float], action: Sequence[float], reward: float, next_state: Sequence[float],
             done: bool) -> None:
        """
        Adds a new memory sample
        :param state: the previous state
        :param action: the action that was taken
        :param reward: the reward for the action
        :param next_state: the current state
        :param done: if the actor has reached a termination state
        """
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[List[Sequence[float]], List[Sequence[float]],
                                               List[float], List[Sequence[float]], List[bool]]:
        """
        Uniformly samples without replacement a batch size
        :param batch_size:
        :return: a tuple with list state, action reward, new action and done flag
        """
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            if isinstance(action, np.float32):
                action = np.array([np.float64(action)])
            action_batch.append(action)

            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def clear(self) -> None:
        """
        Clears the memory buffer
        """
        self.buffer.clear()

    def __len__(self) -> int:
        """
        Returns the current memory size
        :return: the current memory size
        """
        return len(self.buffer)


class KDTreeMemory(Memory):
    """
    An experience replay implementation that hopes to reduce class imbalance by reducing the number of very similar
    data points by using a KD Tree
    """

    def __init__(self, max_size: int, n_similar: int = 100, dim: int = 4) -> None:
        """
        Initializes the KD tree
        :param max_size: the maximum number of points in the KD tree
        :param n_similar: the number of similar data points before points start getting deleted
        :param dim: the dimensions of the data-points added
        """
        super().__init__(max_size)
        self.n_similar = n_similar
        self.dim = 4

        self.tree = kdtree.create(dimensions=dim)

    def push(self, state: Sequence[float], action: Sequence[float], reward: float, next_state: Sequence[float],
             done: bool) -> None:
        """
        Normalizes the state vector and tries to search in the tree for a similar state, if there are
        already n_similar number then the data-point is ignored. Otherwise it is added

        :param state: the previous state
        :param action: the action that was taken
        :param reward: the reward for the action
        :param next_state: the current state
        :param done: if the actor has reached a termination state
        """
        target, dis, theta_lookahead, theta_recovery, theta_near = state
        n = [np.abs(dis), np.abs(theta_lookahead) / np.pi, np.abs(theta_recovery) / np.pi, np.abs(theta_near) / np.pi]

        out = self.tree.search_nn_dist(n, .025)

        submit = len(out) <= self.n_similar

        if submit:
            self.tree.add(n)
            super(KDTreeMemory, self).push(state, action, reward, next_state, done)
        # else:
        #     dists = np.linalg.norm(np.array(n) - np.array(out), axis=1) ** 2
        #     print(n, dists)

    def clear(self) -> None:
        """
        Clears the KD Tree by recreating a new one from scratch
        """
        self.tree = kdtree.create(dimensions=self.dim)
        super(KDTreeMemory, self).clear()
