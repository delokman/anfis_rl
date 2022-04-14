from typing import List

import torch
from torch import nn

from anfis.joint_membership import _mk_param
from new_ddpg.input_membership import JointMembership


class SymmetricCenterOfMaximum(JointMembership):
    def left_x(self):
        pass

    def right_x(self):
        pass

    def __init__(self, center: float, output_poses: List[float], constant_center=True):
        super().__init__(len(output_poses) * 2 + 1)

        if constant_center:
            self.register_buffer("center", torch.tensor([center]))
        else:
            self.register_parameter('center', _mk_param(center))

        self.register_parameter("log_weights", nn.Parameter(torch.log(torch.tensor(output_poses))))

    def forward(self, _):
        weights = self.log_weights.exp()
        total = torch.cumsum(weights, 0)

        return torch.concat((self.center - total.flip([0]), self.center, self.center + total))


class CenterOfMaximum(JointMembership):
    def left_x(self):
        pass

    def right_x(self):
        pass

    def __init__(self, output_poses: List[float]):
        super().__init__(len(output_poses))

        self.register_parameter("log_weights", nn.Parameter(torch.log(torch.tensor(output_poses))))

    def forward(self, _):
        weights = self.log_weights.exp()
        total = torch.cumsum(weights, 0)

        return total
