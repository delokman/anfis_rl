from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from new_ddpg.input_membership import JointTrapMembership, JointMembership


class JointFuzzifyLayer(torch.nn.Module):
    def __init__(self, membership_fncs: List[JointMembership]):
        super(JointFuzzifyLayer, self).__init__()

        self.register_buffer("num_outputs", torch.tensor([mem.num_outputs for mem in membership_fncs]))

        res = self.num_outputs.cumsum(0).roll(1, 0)
        res[0] = 0

        self.register_buffer("start_indexes", res)
        self.register_buffer("len", torch.tensor(len(membership_fncs)))
        self.membership_fncs = nn.ModuleList(membership_fncs)

    def forward(self, x):
        output = []

        for i, membership in enumerate(self.membership_fncs):
            out = membership(x[:, i:i + 1])
            output.append(out)

        return torch.cat(output, dim=1)


class JointAnfisNet(nn.Module):
    def __init__(self, input_variables,
                 output_variables,
                 mamdani_ruleset=None):
        super(JointAnfisNet, self).__init__()
        self.input_variables = input_variables
        self.output_variables = output_variables

        self.fuzzify = JointFuzzifyLayer(input_variables)

        self.fuzzify = torch.jit.trace_module(self.fuzzify, {"forward": torch.randn((1000, 4))})

    def set_training_mode(self, mode):
        self.train(mode)

    def is_cuda(self) -> bool:
        return next(self.parameters()).is_cuda

    def plot_fuzzified(self, x, fuzzyfied):
        with torch.no_grad():
            Y = fuzzyfied.cpu().detach()
            x = x.cpu().detach()

            for i in range(self.fuzzify.len):
                fig, ax = plt.subplots()

                start = self.fuzzify.start_indexes[i]
                end = start + self.fuzzify.num_outputs[i]

                ax.plot(x[:, i], Y[:, start:end])

            plt.show()

    def forward(self, x):
        fuzzyfied = self.fuzzify(x)
        # self.plot_fuzzified(x, fuzzyfied)

        y_pred = None

        return y_pred


if __name__ == '__main__':
    mem1 = JointTrapMembership(0, np.log(np.array([1., 1., 1.])))
    mem2 = JointTrapMembership(0, np.log(np.array([1., 1.])))

    a = JointAnfisNet([mem1, mem2], None, None)

    B = 1000
    E = 5
    x = torch.linspace(-5, 5, B)
    x = x.repeat(E, 1).T

    out = a(x)

    print(out)
