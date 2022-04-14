import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn, profiler

from new_ddpg.input_membership import JointTrapMembership
from new_ddpg.output_membership import CenterOfMaximum, SymmetricCenterOfMaximum
from rl.predifined_anfis import dist_target_dist_per_theta_lookahead_theta_far_theta_near_with_vel
from vizualize.auto_grad_viz import make_dot


class JointAnfisNet(nn.Module):
    def __init__(self, input_variables,
                 output_variables,
                 mamdani_ruleset):
        super(JointAnfisNet, self).__init__()
        self.input_variables = input_variables
        self.output_variables = output_variables

        # FUZZIFY
        self.register_buffer("num_inputs", torch.tensor([mem.num_outputs for mem in input_variables]))

        res = self.num_inputs.cumsum(0).roll(1, 0)
        res[0] = 0

        self.register_buffer("start_indexes", res)
        self.register_buffer("len", torch.tensor(len(input_variables)))
        self.membership_fncs = nn.ModuleList(input_variables)

        # OUTPUT
        self.output_membership = nn.ModuleList(output_variables)

        self.register_buffer("num_outputs", torch.tensor([mem.num_outputs for mem in output_variables]))

        res = self.num_outputs.cumsum(0).roll(1, 0)
        res[0] = 0

        self.register_buffer("start_output_indexes", res)

        # RULES
        input_rules, output_rules = self.convert_rules(mamdani_ruleset)
        self.register_buffer("input_rules", input_rules)
        self.register_buffer("output_rules", output_rules)

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
                end = start + self.num_inputs[i]

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
        start_indexes = torch.cumsum(self.num_inputs - 1, dim=0).roll(1, 0)
        start_indexes[0] = 0

        linguistic_index = torch.tensor(rules['variable_rule_index'])
        membership_indices = torch.tensor(rules['membership_indices'])

        start_indexes = start_indexes.repeat(linguistic_index.shape[0], 1)
        input_indexes = torch.gather(start_indexes, 1, linguistic_index) + membership_indices

        start_output_indexes = torch.cumsum(self.num_outputs, dim=0).roll(1, 0)
        start_output_indexes[0] = 0

        outputs_membership = torch.tensor(rules['outputs_membership'])
        outputs_membership_velocity = torch.tensor(rules['outputs_membership_velocity'])

        output_rules = torch.cat((outputs_membership, outputs_membership_velocity), dim=1) + start_output_indexes

        return input_indexes, output_rules

    @staticmethod
    def t_norm(weights):
        return torch.min(weights, dim=2)[0]

    def rules(self, fuzzified):

        weights = fuzzified[:, self.input_rules]
        return self.t_norm(weights)

    def output_weights(self, weights):
        output = []

        for membership in self.output_membership:
            out = membership(weights)
            output.append(out)

        return torch.cat(output, dim=0)

    def defuzzify(self, normalized_weights, output_weights):
        output_weights = output_weights[self.output_rules]

        return torch.mm(normalized_weights, output_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fuzzyfied: torch.Tensor = self.fuzzify(x)

        weights = self.rules(fuzzyfied)

        normalized_weights = F.normalize(weights, p=1, dim=1)

        output_weights = self.output_weights(normalized_weights)
        defuzzify = self.defuzzify(normalized_weights, output_weights)

        return defuzzify


def profile():
    mem1 = JointTrapMembership(0, np.log(np.array([1., 1.])))
    mem2 = JointTrapMembership(0, np.log(np.array([1., 1., 1., 1, 1])))
    mem3 = JointTrapMembership(0, np.log(np.array([1., 1., 1])))
    mem4 = JointTrapMembership(0, np.log(np.array([1., 1., 1])))
    mem5 = JointTrapMembership(0, np.log(np.array([1., 1., 1])))

    ruleset = dist_target_dist_per_theta_lookahead_theta_far_theta_near_with_vel()

    out1 = SymmetricCenterOfMaximum(0., [1., 1., 1., 1.])
    out2 = CenterOfMaximum([0.1, 1., 2.])

    anfis = JointAnfisNet([mem1, mem2, mem3, mem4, mem5], [out1, out2], ruleset)
    # fuzzify = torch.jit.trace_module(a, {"forward": torch.randn((1000, 4))})

    B = 1
    E = 5
    x = torch.linspace(-5, 5, B)
    x = x.repeat(E, 1).T

    x = x.to('cuda')
    anfis = anfis.to('cuda')

    traced_anfis = torch.jit.trace(anfis, x)

    o = []
    o2 = []
    y = []

    print("Starting")

    N = 1000

    for power in range(1, 10):
        with profiler.profile() as prof:
            for _ in range(N):
                anfis(x)
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))

        diff1 = prof.key_averages().total_average().self_cpu_time_total / N
        cuda1 = prof.key_averages().total_average().self_cuda_time_total / N

        with profiler.profile() as prof:
            for _ in range(N):
                traced_anfis(x)
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))

        diff2 = prof.key_averages().total_average().self_cpu_time_total / N
        cuda2 = prof.key_averages().total_average().self_cuda_time_total / N

        if power != 1:
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


def main_test():
    mem1 = JointTrapMembership(0, np.log(np.array([1., 1.])))
    mem2 = JointTrapMembership(0, np.log(np.array([1., 1., 1., 1, 1])))
    mem3 = JointTrapMembership(0, np.log(np.array([1., 1., 1])))
    mem4 = JointTrapMembership(0, np.log(np.array([1., 1., 1])))
    mem5 = JointTrapMembership(0, np.log(np.array([1., 1., 1])))

    ruleset = dist_target_dist_per_theta_lookahead_theta_far_theta_near_with_vel()

    out1 = SymmetricCenterOfMaximum(0., [1., 1., 1., 1.])
    out2 = CenterOfMaximum([0.1, 1., 2.])

    anfis = JointAnfisNet([mem1, mem2, mem3, mem4, mem5], [out1, out2], ruleset)
    anfis.cuda()
    # fuzzify = torch.jit.trace_module(a, {"forward": torch.randn((1000, 4))})

    B = 1000
    E = 5
    x = torch.linspace(-5, 5, B)
    x = x.repeat(E, 1).T
    x = x.to('cuda')

    traced_anfis = torch.jit.trace(anfis, x)

    print(traced_anfis.forward.graph)
    print(traced_anfis.forward.code)

    out = anfis(x)

    out = out.sum()
    g = make_dot(out, dict(anfis.named_parameters()))
    g.view()

    print(out)


if __name__ == '__main__':
    profile()
