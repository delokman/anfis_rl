from abc import abstractmethod, ABC
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import profiler

from anfis.joint_membership import _mk_param
from new_ddpg import FLOAT_TORCH_TYPE


class JointMembership(ABC, torch.nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs

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
        super().__init__(len(trap_widths) + 1)

        if constant_center:
            self.register_buffer("center", torch.tensor(center, dtype=FLOAT_TORCH_TYPE))
        else:
            self.register_parameter('center', _mk_param(center, dtype=FLOAT_TORCH_TYPE))

        self.register_parameter("log_weights", torch.nn.Parameter(torch.tensor(trap_widths, dtype=FLOAT_TORCH_TYPE)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.log_weights.exp()
        total = torch.cumsum(weights, 0)

        total = torch.concat((self.center - total.flip([0]), self.center + total))

        x1 = total[:-1:2]
        x2 = total[1::2]

        up_slope = (x - x1) / (x2 - x1)
        down_slope = 1 - up_slope
        torch.clamp_(up_slope, min=0, max=1)
        torch.clamp_(down_slope, min=0, max=1)

        out = torch.ones((x.shape[0], self.num_outputs), device=x.device)

        out[:, 1:] = up_slope
        out[:, :-1] *= down_slope

        return out


def test():
    weights = torch.log(torch.tensor([[1], [2], [3], [4.]], requires_grad=True, device='cuda'))

    c = torch.tensor(0.0, device=weights.device, requires_grad=True)

    with torch.no_grad():
        a = torch.cumsum(weights.exp(), dim=0)[-1] * 1.5
        a = a.item()
        x = torch.linspace(c - a, c + a, steps=1)
        x = x.repeat(5, 1)
        x = x.to(c.device)

    trapezoid = JointTrapMembership(0, np.log(np.array([1, 2, 3, 4])))
    trapezoid.cuda()
    funcs = {'forward': x}
    trapezoid_opt = torch.jit.trace_module(trapezoid, funcs)

    o = []
    o2 = []
    y = []

    print("Starting")

    for power in range(1, 65):
        with torch.no_grad():
            a = torch.cumsum(weights.exp(), dim=0)[-1] * 1.5
            a = a.item()
            x = torch.linspace(c - a, c + a, steps=power)
            x = x.repeat(5, 1)
            x = x.to(c.device)

        with profiler.profile() as prof:
            for _ in range(100):
                trapezoid(x)
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))

        diff1 = prof.key_averages().total_average().self_cpu_time_total
        cuda1 = prof.key_averages().total_average().self_cuda_time_total

        with profiler.profile() as prof:
            for _ in range(100):
                trapezoid_opt(x)
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))

        diff2 = prof.key_averages().total_average().self_cpu_time_total
        cuda2 = prof.key_averages().total_average().self_cuda_time_total

        o.append((diff1, diff2))
        o2.append((cuda1, cuda2))

        y.append(power)
    o = np.array(o)
    plt.plot(y, o[:, 0], label='normal')
    plt.plot(y, o[:, 1], label='optimal')
    plt.legend()
    plt.show()

    o = np.array(o2)
    plt.plot(y, o[:, 0], label='normal')
    plt.plot(y, o[:, 1], label='optimal')
    plt.legend()
    plt.show()


def plotted():
    weights = np.array([1., 2, 3])
    c = 0

    with torch.no_grad():
        a = np.cumsum(weights, axis=0)[-1] * 1.5
        temp_c = c
        x = torch.linspace(temp_c - a, temp_c + a, steps=1000)
        x = x.reshape((-1, 1))

    trapezoid = JointTrapMembership(c, np.log(weights), constant_center=False)
    trapezoid.cuda()
    x = x.cuda()

    Y = trapezoid(x)
    crit = torch.nn.MSELoss()

    zero = Y - 0.1
    zero = zero.detach()

    loss: torch.Tensor = crit(Y, zero)
    loss.backward()

    print(loss)
    print(trapezoid.center.grad)

    Y = Y.cpu().detach()
    x = x.cpu().detach()

    plt.plot(x, Y)
    plt.title('Pytorch Trapezoid Function')
    plt.show()


if __name__ == '__main__':
    plotted()
