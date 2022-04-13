import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn

from new_ddpg.input_membership import JointTrapMembership
from rl.predifined_anfis import dist_target_dist_per_theta_lookahead_theta_far_theta_near_with_vel


class JointAnfisNet(nn.Module):
    def __init__(self, input_variables,
                 output_variables,
                 mamdani_ruleset):
        super(JointAnfisNet, self).__init__()
        self.input_variables = input_variables
        self.output_variables = output_variables

        # FUZZIFY
        self.register_buffer("num_outputs", torch.tensor([mem.num_outputs for mem in input_variables]))

        res = self.num_outputs.cumsum(0).roll(1, 0)
        res[0] = 0

        self.register_buffer("start_indexes", res)
        self.register_buffer("len", torch.tensor(len(input_variables)))
        self.membership_fncs = nn.ModuleList(input_variables)

        # RULES
        input_rules, output_rules = self.convert_rules(mamdani_ruleset)

        self.register_buffer("input_rules", input_rules)

    def fuzzify(self, x):
        output = []

        for i, membership in enumerate(self.membership_fncs):
            out = membership(x[:, i:i + 1])
            output.append(out)

        return torch.cat(output, dim=1)

    def set_training_mode(self, mode):
        self.train(mode)

    def is_cuda(self) -> bool:
        return next(self.parameters()).is_cuda

    @torch.jit.ignore
    def plot_fuzzified(self, x, fuzzyfied):
        with torch.no_grad():
            Y = fuzzyfied.cpu().detach()
            x = x.cpu().detach()

            for i in range(self.len):
                fig, ax = plt.subplots()

                start = self.start_indexes[i]
                end = start + self.num_outputs[i]

                ax.plot(x[:, i], Y[:, start:end])

            plt.show()

    @torch.jit.ignore
    def plot_weights(self, x, weights):
        with torch.no_grad():
            Y = weights.cpu().detach()
            x = x.cpu().detach()

            fig, ax = plt.subplots()

            ax.plot(x[:, 0], Y)

            plt.show()

    def convert_rules(self, rules):
        start_indexes = torch.cumsum(self.num_outputs - 1, dim=0).roll(1, 0)
        start_indexes[0] = 0

        linguistic_index = torch.tensor(rules['variable_rule_index'])
        membership_indices = torch.tensor(rules['membership_indices'])

        start_indexes = start_indexes.repeat(linguistic_index.shape[0], 1)

        outputs_membership = torch.tensor(rules['outputs_membership'])
        outputs_membership_velocity = torch.tensor(rules['outputs_membership_velocity'])

        input_indexes = torch.gather(start_indexes, 1, linguistic_index) + membership_indices

        output_rules = None

        return input_indexes, output_rules

    def t_norm(self, weights):
        return torch.min(weights, dim=2)[0]

    def rules(self, fuzzified):

        weights = fuzzified[:, self.input_rules]
        return self.t_norm(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fuzzyfied: torch.Tensor = self.fuzzify(x)
        # self.plot_fuzzified(x, fuzzyfied)

        weights = self.rules(fuzzyfied)

        normalized_weights = F.normalize(weights, p=1, dim=1)

        # self.plot_weights(x, weights)

        return weights


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
