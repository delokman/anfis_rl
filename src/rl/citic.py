import torch
import torch.nn.functional as F
from torch import nn


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, fcs1_units=32, fc2_units=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_size, fcs1_units)
        self.linear2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.linear3 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.elu(self.linear1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.elu(self.linear2(x))
        return self.linear3(x)
