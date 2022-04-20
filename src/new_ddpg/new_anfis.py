from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn, profiler

from new_ddpg.input_membership import JointTrapMembership, JointMembership
from new_ddpg.output_membership import CenterOfMaximum, SymmetricCenterOfMaximum
from rl.predifined_anfis import dist_target_dist_per_theta_lookahead_theta_far_theta_near_with_vel
from vizualize.auto_grad_viz import make_dot
from new_ddpg import FLOAT_TORCH_TYPE


class JointAnfisNet(nn.Module):
    def __init__(self, input_variables,
                 output_variables,
                 mamdani_ruleset, tanh_centers: List[float], tanh_scale: List[float]):
        super(JointAnfisNet, self).__init__()
        self.register_buffer("output_scaling", torch.tensor(tanh_scale, dtype=FLOAT_TORCH_TYPE))
        self.register_buffer("output_bias", torch.tensor(tanh_centers, dtype=FLOAT_TORCH_TYPE))

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
    def plot_weights(self, x, weights):
        with torch.no_grad():
            Y = weights.cpu().detach()
            x = x.cpu().detach()

            fig, ax = plt.subplots()

            ax.plot(x[:, 0], Y)

            plt.show()

    def convert_rules(self, rules):
        start_indexes = torch.cumsum(self.num_inputs, dim=0).roll(1, 0)
        start_indexes[0] = 0

        linguistic_index = torch.tensor(rules['variable_rule_index'])
        membership_indices = torch.tensor(rules['membership_indices'])

        start_indexes = start_indexes.repeat(linguistic_index.shape[0], 1)
        input_indexes = torch.gather(start_indexes, 1, linguistic_index) + membership_indices

        start_output_indexes = torch.cumsum(self.num_outputs, dim=0).roll(1, 0)
        start_output_indexes[0] = 0

        outputs_membership = torch.tensor(rules['outputs_membership'])
        outputs_membership = torch.max(outputs_membership) - outputs_membership

        outputs_membership_velocity = torch.tensor(rules['outputs_membership_velocity'])

        output_rules = torch.cat((outputs_membership, outputs_membership_velocity), dim=1) + start_output_indexes

        return input_indexes, output_rules

    @staticmethod
    def t_norm(weights):
        return torch.min(weights, dim=2)[0]

    def rules(self, fuzzified):

        weights = fuzzified[:, self.input_rules]
        return self.t_norm(weights)

    def get_output_weights(self, weights):
        output = []

        for membership in self.output_membership:
            out = membership(weights)
            output.append(out)

        return torch.cat(output, dim=0)

    def defuzzify(self, normalized_weights, output_weights):
        output_weights = output_weights[self.output_rules]

        return torch.mm(normalized_weights, output_weights)

    @property
    def device(self) -> torch.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        :return:"""
        for param in self.parameters():
            return param.device
        return torch.device('cpu')

    def plot_fuzzified(self, data=None, fuzzified=None):
        with torch.no_grad():
            last_num = 0

            for i, membership in enumerate(self.membership_fncs):
                membership: JointMembership

                x_min = membership.left_x()
                x_max = membership.right_x()

                diff = (x_max - x_min) * 0.1

                x = torch.linspace(membership.left_x() - diff, membership.right_x() + diff, steps=100,
                                   device=self.device).reshape((-1, 1))

                out = membership(x)

                fig, ax = plt.subplots()

                if data is not None and fuzzified is not None:
                    x_d = data[:, i].repeat((membership.num_outputs, 1)).T
                    next_num = last_num + membership.num_outputs

                    out_d = fuzzified[:, last_num:next_num]

                    last_num = next_num
                    ax.scatter(x_d.cpu(), out_d.cpu())

                ax.plot(x.cpu(), out.cpu())
                plt.show()
                plt.close(fig)

    def print_rules(self, fuzzified, weights, output_weights):
        poses = ['distance_target is Close', "distance_target is Far", "distance_line is Left Edge",
                 "distance_line is Left", "distance_line is Close Left", "distance_line is Center",
                 "distance_line is Close Right", "distance_line is Right", "distance_line is Right Edge",
                 'theta_lookahead is Left Edge', "theta_lookahead is Left", "theta_lookahead is Center",
                 "theta_lookahead is Right", "theta_lookahead is Right Edge", 'theta_far is Left Edge',
                 "theta_far is Left", "theta_far is Center", "theta_far is Right", "theta_far is Right Edge",
                 'theta_near is Left Edge', "theta_near is Left", "theta_near is Center", "theta_near is Right",
                 "theta_near is Right Edge",
                 ]

        outs = ['angular_vel is Very Hard Left', 'angular_vel is Hard Left', 'angular_vel is Left',
                'angular_vel is Soft Left', 'angular_vel is Zero', 'angular_vel is Soft Right', 'angular_vel is Right',
                'angular_vel is Hard Right', 'angular_vel is Very Hard Right', "vel is Slow", 'vel is Medium',
                'vel is Fast']

        data = []
        n_outs = self.output_rules.shape[1]

        for r in range(self.input_rules.shape[0]):
            row = self.input_rules[r]

            for c in range(row.shape[0]):
                index = row[c].item()
                data.append(poses[index])
                a = fuzzified[0, index].item()
                data.append(" ")
                data.append(a)
                data.append(" AND ")

            data = data[:-1]
            data.append(" IS ")
            data.append(weights[0, r].item())
            data.append(" THEN ")

            row = self.output_rules[r]
            for c in range(n_outs):
                index = row[c].item()
                data.append(outs[index])
                data.append(" ")
                a = output_weights[index].item()
                data.append(a)
                data.append(" AND ")

            data = data[:-1]
            data.append('\n')

        print("".join(map(str, data)))

    def constrain_scaling(self, defuzzify: torch.Tensor):
        contained = torch.tanh(defuzzify)

        scaled_constraint = contained * self.output_scaling + self.output_bias
        return scaled_constraint

    def plot_outputs(self):
        with torch.no_grad():
            zero = torch.zeros(1, device=self.device)

            for membership in self.output_membership:
                out = membership(zero).cpu()

                fig, ax = plt.subplots()

                for i in out:
                    ax.plot([i - 1, i, i + 1], [0, 1, 0])

            plt.show()
            plt.close(fig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.fuzzyfied: torch.Tensor = self.fuzzify(x)

        # self.plot_fuzzified(x, self.fuzzyfied)

        # self.fuzzyfied_data(x, self.fuzzyfied)

        self.weights = self.rules(self.fuzzyfied)

        # self.print_rules(self.fuzzyfied, self.weights)

        self.normalized_weights = F.normalize(self.weights, p=1, dim=1)

        self.output_weights = self.get_output_weights(self.normalized_weights)
        self.defuzzify_out = self.defuzzify(self.normalized_weights, self.output_weights)

        self.constrained = self.constrain_scaling(self.defuzzify_out)

        return self.constrained

    def set_training_mode(self, mode):
        self.train(mode)


def profile():
    mem1 = JointTrapMembership(0, np.log(np.array([1.])))
    mem2 = JointTrapMembership(0, np.log(np.array([1., 1., 1., 1, 1, 1])))
    mem3 = JointTrapMembership(0, np.log(np.array([1., 1., 1, 1])))
    mem4 = JointTrapMembership(0, np.log(np.array([1., 1., 1, 1])))
    mem5 = JointTrapMembership(0, np.log(np.array([1., 1., 1, 1])))

    ruleset = dist_target_dist_per_theta_lookahead_theta_far_theta_near_with_vel()

    out1 = SymmetricCenterOfMaximum(0., [1., 1., 1., 1.])
    out2 = CenterOfMaximum([0, 1., 1])

    anfis = JointAnfisNet([mem1, mem2, mem3, mem4, mem5], [out1, out2], ruleset, [4, 2], [-4, 0])
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


def min_max_num_trapezoids(min_v, max_v, num):
    c = (min_v + max_v) / 2

    num = num - 1

    num = np.full(num, (max_v - c) / num)
    num[0] /= 2

    return c, np.log(num)


def min_max_num_center_of_max(min_v, max_v, num):
    steps = (max_v - min_v) / (num - 1)

    num = np.full(num, steps)

    if min_v <= 0:
        min_v = 1e-6

    num[0] = min_v

    return num


def min_max_num_symmetric_center_of_max(min_v, max_v, num):
    c = (max_v + min_v) / 2

    num = (num - 1) // 2

    steps = (max_v - c) / num

    num = np.full(num, steps)

    return c, num


def main_test():
    mem1 = JointTrapMembership(*min_max_num_trapezoids(0, 1, 2))
    mem2 = JointTrapMembership(*min_max_num_trapezoids(-1, 1, 7))
    mem3 = JointTrapMembership(*min_max_num_trapezoids(-np.pi, np.pi, 5))
    mem4 = JointTrapMembership(*min_max_num_trapezoids(-np.pi, np.pi, 5))
    mem5 = JointTrapMembership(*min_max_num_trapezoids(-np.pi, np.pi, 5))

    ruleset = dist_target_dist_per_theta_lookahead_theta_far_theta_near_with_vel()

    out1 = SymmetricCenterOfMaximum(*min_max_num_symmetric_center_of_max(-4, 4, 9))
    out2 = CenterOfMaximum(min_max_num_center_of_max(0, 2, 3))

    anfis = JointAnfisNet([mem1, mem2, mem3, mem4, mem5], [out1, out2], ruleset, [4, 2], [-4, 0])
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
