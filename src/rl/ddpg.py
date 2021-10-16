import copy
import pathlib

import numpy as np
import torch
from torch import optim

from anfis.utils import save_fuzzy_membership_functions
from rl.citic import Critic
from rl.memory import Memory
from rl.prioritized_memory_replay import PrioritizedReplayBuffer


class DDPGAgent(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, anf, hidden_size=32, actor_learning_rate=1e-4,
                 critic_learning_rate=1e-5, gamma=0.99, tau=1e-3, max_memory_size=50000, priority=True, grad_clip=1):
        # Params
        super().__init__()
        self.grad_clip = grad_clip
        self.use_cuda = torch.cuda.is_available()
        self.use_cuda = False

        self.num_states = num_inputs
        self.num_actions = num_outputs
        self.gamma = gamma
        self.tau = tau
        self.curr_states = np.zeros(num_inputs)
        # Networks
        self.actor = anf
        self.actor_target = copy.deepcopy(anf)

        self.input_params = {
            'num_in': num_inputs,
            'num_out': num_outputs,
            'num_hidden': hidden_size,
            'actor_lr': actor_learning_rate,
            'critic_lr': critic_learning_rate,
            'gamma': gamma,
            'tau': tau,
            'max_memory': max_memory_size
        }

        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.priority = priority
        if priority:
            self.memory = PrioritizedReplayBuffer(max_memory_size, .5)
        else:
            self.memory = Memory(max_memory_size)

        self.critic_criterion = torch.nn.MSELoss()
        self.actor_optimizer = optim.SGD(self.actor.parameters(), lr=actor_learning_rate, momentum=0.99)
        self.critic_optimizer = optim.SGD(self.critic.parameters(), lr=critic_learning_rate, momentum=0.99)

        for name, v in self.actor_optimizer.defaults.items():
            self.input_params[f'actor_optim_{name}'] = v

        for name, v in self.critic_optimizer.defaults.items():
            self.input_params[f'critic_optim_{name}'] = v

        if self.use_cuda:
            self.actor.cuda()
            # self.actor.cuda_memberships()
            # self.actor.cuda_mamdani()

            self.actor_target.cuda()
            # self.actor_target.cuda_memberships()
            # self.actor_target.cuda_mamdani()

            self.critic.cuda()
            self.critic_target.cuda()

        self.ordered_dict = torch.nn.ModuleDict()
        self.ordered_dict['actor'] = self.actor
        self.ordered_dict['actor_target'] = self.actor_target
        self.ordered_dict['critic'] = self.critic
        self.ordered_dict['critic_target'] = self.critic_target

    def save_checkpoint(self, location):
        state_dicts = {
            'actor': self.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }

        loc = pathlib.Path(location)
        epoch = int(loc.name[:-len('.chkp')].split('-')[0])
        folder = 'mfs'

        mfs_folder = loc.parent.joinpath(folder)
        mfs_folder.mkdir(exist_ok=True)
        mfs_checkpoint = mfs_folder.joinpath(f'{epoch}.txt')
        save_fuzzy_membership_functions(self.actor, mfs_checkpoint)

        torch.save(state_dicts, location)

    def load_checkpoint(self, location):
        state_dicts = torch.load(location)

        self.load_state_dict(state_dicts['actor'])
        self.actor_optimizer.load_state_dict(state_dicts['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dicts['critic_optimizer'])

    def get_action(self, state):
        state = torch.tensor(state, requires_grad=True, dtype=torch.float32).unsqueeze(0)
        #        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        if self.use_cuda:
            state = state.cuda()

        action = self.actor.forward(state)

        if self.use_cuda:
            action = action.cpu().detach().numpy()[0, 0]
        else:
            action = action.detach().numpy()[0, 0]
        return action

    def update(self, batch_size):
        if self.priority:
            states, actions, rewards, next_states, _, weights, batch_idxes = self.memory.sample(batch_size, 0.5)
        else:
            states, actions, rewards, next_states, _ = self.memory.sample(batch_size, 0)
            weights, batch_idxes = np.ones_like(rewards), None

        states = torch.FloatTensor(states)
        # print(actions)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        actions = torch.reshape(actions, (batch_size, 1))
        weights = torch.FloatTensor(weights)

        if self.use_cuda:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_states = next_states.cuda()

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals * weights, Qprime * weights)

        # Actor loss
        policy_loss = self.critic.forward(states, self.actor.forward(states)).mean()
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()

        # g = make_dot(policy_loss, dict(self.named_parameters()), None)
        # g.view()
        # sys.exit()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # update target networks
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

        if self.priority:
            TD_error = torch.abs(Qprime - Qvals) + 1e-6

            self.memory.update_priorities(batch_idxes, TD_error)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
