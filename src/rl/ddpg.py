import copy

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable

from rl.citic import Critic
from rl.memory import Memory


class DDPGAgent:
    def __init__(self, num_inputs, num_outputs, anf, hidden_size=128, actor_learning_rate=1e-3,
                 critic_learning_rate=1e-4, gamma=0.99, tau=1e-3, max_memory_size=50000):
        # Params
        self.num_states = num_inputs
        self.num_actions = num_outputs
        self.gamma = gamma
        self.tau = tau
        self.curr_states = np.array([0, 0, 0])
        # Networks
        self.actor = anf
        self.actor_target = copy.deepcopy(anf)

        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = Memory(max_memory_size)
        self.critic_criterion = torch.nn.MSELoss(reduction='sum')
        self.actor_optimizer = optim.SGD(self.actor.parameters(), lr=1e-6 * 7, momentum=0.99)
        self.critic_optimizer = optim.SGD(self.critic.parameters(), lr=critic_learning_rate, momentum=0.99)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0, 0]
        return action

    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        # print(actions)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        actions = torch.reshape(actions, (batch_size, 1))

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime) / 5.
        if critic_loss.item() > 20:
            critic_loss = critic_loss / critic_loss.item() * 20.0

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean() / -10.
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
