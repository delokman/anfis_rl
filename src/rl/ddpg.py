import copy
import pathlib
from typing import Sequence, Union, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from anfis.utils import save_fuzzy_membership_functions
from rl.citic import Critic
from rl.memory import Memory
from rl.prioritized_memory_replay import PrioritizedReplayBuffer


class DDPGAgent(torch.nn.Module):
    """
    Custom implementation of the classic DDPG (Deep Deterministic Policy Gradient) algorithm,
    in order to optimize a continuous actions state based on a reward function. More details can
    be found https://keras.io/examples/rl/ddpg_pendulum/ or https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html
    """

    def __init__(self, num_inputs: int, num_outputs: int, anf: torch.nn.Module, hidden_size: int = 32,
                 actor_learning_rate: float = 1e-4,
                 critic_learning_rate: float = 1e-5, gamma: float = 0.99, tau: float = 1e-3,
                 max_memory_size: int = 50000, priority: bool = True, grad_clip: float = 1,
                 alpha: Optional[float] = 0.9, beta: Optional[float] = 0.9) -> None:
        """
        Initializes the DDPG agent

        :param num_inputs: The dimension size of the input state
        :param num_outputs: The dimension size of the output of the actor
        :param anf: The actor, in this case the ANFIS system
        :param hidden_size: the size of the hidden layer for the CNN
        :param actor_learning_rate: the actor learning rate for the optimizer
        :param critic_learning_rate: the critic learning rate for to optimize, should be larger than the actor
        learning rate to improve stability
        :param gamma: the discount factor for the previous Q value state
        :param tau: the soft update coefficient ("Polyak update", between 0 and 1) used for the target network update
        :param max_memory_size: the number of previous states to keep in memory as possible training samples
        :param priority: if True use priority experience replay instead of the uniform experience replay sampling method
        :param grad_clip: Sometimes to improve stability the gradients of the gradients need to be clipped in order to
        smooth the update function improve stability
        :param alpha: for the priority experience replay only, how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        :param beta: for the priority experience replay only, To what degree to use importance weights (0 - no corrections, 1 - full correction)
        """
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
            'max_memory': max_memory_size,
            'priority': priority,
            'grad_clip': grad_clip,
        }

        self.critic = Critic(self.num_states, self.num_actions)
        self.critic_target = Critic(self.num_states, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.priority = priority
        if priority:
            self.memory = PrioritizedReplayBuffer(max_memory_size, alpha, beta)
            self.input_params['mem_alpha'] = alpha
            self.input_params['mem_beta'] = beta
        else:
            self.memory = Memory(max_memory_size)

        self.critic_criterion = torch.nn.MSELoss(reduction='mean')
        self.actor_optimizer = optim.RAdam(self.actor.parameters(), lr=actor_learning_rate)
        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.RAdam(self.critic.parameters(), lr=critic_learning_rate)
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        param_type = [int, float, str, bool, torch.Tensor]

        for name, v in self.actor_optimizer.defaults.items():
            if type(v) in param_type:
                if isinstance(v, torch.Tensor):
                    v = v.detach()

                self.input_params[f'actor_optim_{name}'] = v

        for name, v in self.critic_optimizer.defaults.items():
            if type(v) in param_type:
                if isinstance(v, torch.Tensor):
                    v = v.detach()

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

        self.summary_index = 0

        self._train_inputs = True
        self._train_velocity = True
        self._train_angular = True

        for p in self.actor_target.parameters():
            p.requires_grad = False

        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.compare_models(self.actor, self.actor_target)
        self.compare_models(self.critic, self.critic_target)

    @staticmethod
    def compare_models(model_1: torch.nn.Module, model_2: torch.nn.Module) -> int:
        """
        Function to compare the parameters between two models to make sure they are the same

        :param model_1: the first model to compare
        :param model_2: the second model to compare
        :return: the number of different parameters
        """
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly! :)')

        return models_differ

    @property
    def train_inputs(self) -> bool:
        """
        Gets property for the boolean for enabling and disabling training of the input membership functions
        :return: The training enabled flag
        """
        return self._train_inputs

    @train_inputs.setter
    def train_inputs(self, new_val: bool) -> None:
        """
        Sets property for the boolean for enabling and disabling training of the input membership functions
        :param new_val: Whether the input membership functions should be training or not
        """

        self._train_inputs = new_val

        self.actor.train_inputs = new_val
        self.actor_target.train_inputs = new_val

    @property
    def train_velocity(self) -> bool:
        """
        Get boolean flag on whether the ANFIS velocity output membership function are getting trained
        :return: Velocity output membership training flag
        """
        return self._train_velocity

    @train_velocity.setter
    def train_velocity(self, new_val: bool) -> None:
        """
        Sets the boolean flag on whether the ANFIS velocity output membership function should be training
        :type new_val: Whether to enable training or not
        """

        self._train_velocity = new_val

        self.actor.set_linear_vel_training(new_val)
        self.actor_target.set_linear_vel_training(new_val)

    @property
    def train_angular(self) -> bool:
        """
        Get boolean flag on whether the ANFIS angular velocity output membership function are getting trained
        :return: Angular velocity output membership training flag
        """
        return self._train_angular

    @train_angular.setter
    def train_angular(self, new_val: bool) -> None:
        """
        Sets the boolean flag on whether the ANFIS velocity output membership function should be training
        :type new_val: Whether to enable training or not
        """

        self._train_angular = new_val

        self.actor.set_angular_vel_training(new_val)
        self.actor_target.set_angular_vel_training(new_val)

    def save_checkpoint(self, location: str) -> None:
        """
        Save the current DDPG model to a specific location. Saves the current DDPG state dict, actor optimizer,
        critic optimizer. To a txt file the actor fuzzy membership parameters are also saved.
        :param location: the location to save the checkpoint
        """
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

    def load_checkpoint(self, location: str) -> None:
        """
        Loads the DDPG checkpoint and updates the current state of this DDPG model
        :param location: the checkpoint location
        """
        state_dicts = torch.load(location)

        self.load_state_dict(state_dicts['actor'])
        self.actor_optimizer.load_state_dict(state_dicts['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dicts['critic_optimizer'])

    def get_action(self, state: Sequence[float]) -> Union[float, Tuple[float, float]]:
        """
        Runs the anfis using the current state vector to get the ouput of the system
        :param state: the current state of the system
        :return: the output from the ANFIS, if velocity is not enabled then a single float will be return otherwise a
        tuple of floats will be returned
        """

        # FIXME I do not think requires_grad should be True here
        state = torch.tensor(state, requires_grad=True, dtype=torch.float32).unsqueeze(0)
        #        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        if self.use_cuda:
            state = state.cuda()

        action = self.actor.forward(state)

        if self.use_cuda:
            action = action.cpu().detach().numpy()
        else:
            action = action.detach().numpy()

        if self.actor.velocity:
            return action[0, 0], action[0, 1]
        else:
            return action[0, 0]

    def update(self, batch_size: int, summary: Optional[SummaryWriter] = None) -> None:
        """
        Applies backpropagation on the actor and critic networks in order to maximize the expected reward.
        In depth explanation is located on the DDPG paper: https://arxiv.org/abs/1509.02971
        :param batch_size: The batch size to sample from the experience memory
        :param summary: the tensorboard summary writer to add some debugging information such as the loss values
        """
        if self.priority:
            states, actions, rewards, next_states, _, weights, batch_idxes = self.memory.sample(batch_size)
        else:
            states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
            weights, batch_idxes = np.ones_like(rewards), None

        states = torch.FloatTensor(states)
        # print(actions)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        actions = torch.reshape(actions, (batch_size, self.num_actions))
        weights = torch.FloatTensor(weights)

        if self.use_cuda:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_states = next_states.cuda()

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states).detach()
        next_Q = self.critic_target.forward(next_states, next_actions)
        Qprime = rewards + self.gamma * next_Q
        critic_loss = F.mse_loss(Qvals * weights, Qprime * weights)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = False

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()

        # g = make_dot(policy_loss, dict(self.named_parameters()), None)
        # g.view()
        # sys.exit()

        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        for p in self.critic.parameters():
            p.requires_grad = True

        # update target networks
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

        with torch.no_grad():
            TD_error = (Qprime - Qvals).detach()
        if self.priority:
            self.memory.update_priorities(batch_idxes, torch.abs(TD_error))

        if summary is not None:
            summary.add_scalar("Update/Actor Loss", policy_loss.detach(), global_step=self.summary_index)
            summary.add_scalar("Update/Critic Loss", critic_loss.detach(), global_step=self.summary_index)
            # summary.add_scalar("Update/Velocity Regularization", vel_average, global_step=self.summary_index)
            summary.add_scalar("Update/TD Error", torch.mean(TD_error), global_step=self.summary_index)

            self.summary_index += 1

    def soft_update(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        """
        Apply Polyak update on the target network, otherwise know as a soft update where the target network is updated
        linearly with a proportional `tau` with the source network

        :param target: the target network to update
        :param source: the source network to update towards
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
