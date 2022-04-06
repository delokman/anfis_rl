from abc import abstractmethod, ABC
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt

from anfis.joint_membership import _mk_param
from vizualize.auto_grad_viz import make_dot

torch.autograd.set_detect_anomaly(True)


class JointMembership(ABC, torch.nn.Module):
    @abstractmethod
    def left_x(self):
        pass

    @abstractmethod
    def right_x(self):
        pass


class JointTrapMembership(JointMembership):
    def left_x(self):
        with torch.no_grad():
            return (self.center - torch.sum(self.log_weights.exp())).detach()

    def right_x(self):
        with torch.no_grad():
            return (self.center + torch.sum(self.log_weights.exp())).detach()

    def __init__(self, center: float, trap_widths: List[float], constant_center=True):
        super().__init__()

        if constant_center:
            self.register_buffer("center", torch.tensor(center))
        else:
            self.register_parameter('center', _mk_param(center))

        self.register_parameter("log_weights", torch.nn.Parameter(torch.tensor(trap_widths)[:, None]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.log_weights.exp()
        total = torch.cumsum(weights, 0)

        total = torch.concat((self.center - total.flip([0]), self.center + total))

        x1 = total[:-1:2]
        x2 = total[1::2]

        up_slope = (x[1:] - x1) / (x2 - x1)
        down_slope = 1 - up_slope
        torch.clamp_(up_slope, min=0, max=1)
        torch.clamp_(down_slope, min=0, max=1)

        out = torch.ones_like(x)

        out[1:] = up_slope
        out[:-1] *= down_slope

        return out


if __name__ == '__main__':
    weights = torch.log(torch.tensor([[1], [2], [3], [4.]], requires_grad=True, device='cuda'))

    c = torch.tensor(0.0, device=weights.device, requires_grad=True)

    with torch.no_grad():
        a = torch.cumsum(weights.exp(), dim=0)[-1] * 1.5
        a = a.item()
        x = torch.linspace(c - a, c + a, steps=1000)
        x = x.repeat(5, 1)
        x = x.to(c.device)

    trapezoid = JointTrapMembership(0, np.log(np.array([1, 2, 3, 4])))
    trapezoid.cuda()

    Y = trapezoid(x)

    loss = Y.sum()
    loss.backward()

    dot = make_dot(loss, dict(weight=weights))
    dot.view()

    # trap = torch.jit.trace(trap, example_inputs=(x, c, weights))

    # print(trap.code)
    # print(trap.graph)
    #
    Y = Y.cpu().detach()
    x = x.cpu().detach()

    plt.plot(x.T, Y.T)
    plt.title('Pytorch Trapezoid Function')
    plt.show()
