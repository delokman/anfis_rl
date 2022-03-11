import random
from collections import deque

import kdtree
import numpy as np


class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
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

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class KDTreeMemory(Memory):
    def __init__(self, max_size, n_similar=100, dim=4):
        super().__init__(max_size)
        self.n_similar = n_similar
        self.dim = 4

        self.tree = kdtree.create(dimensions=dim)

    def push(self, state, action, reward, next_state, done):
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

    def clear(self):
        self.tree = kdtree.create(dimensions=self.dim)
        super(KDTreeMemory, self).clear()
