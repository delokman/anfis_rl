from abc import abstractmethod
from collections import OrderedDict

import torch
# from line_profiler_pycharm import profile

from anfis.joint_membership import _mk_param, JointMembership


class JointMembershipHyperOptimized(JointMembership):
    def __init__(self):
        super(JointMembershipHyperOptimized, self).__init__()

    @abstractmethod
    def compute(self, x):
        pass

    # @profile
    def forward(self, x):
        if self.is_cuda and not x.is_cuda:
            x = x.cuda()

        y_pred = self.compute(x)
        torch.clamp_(y_pred, 0, 1)

        if self.padding > 0:
            y_pred = torch.cat([y_pred,
                                torch.zeros(x.shape[0], self.padding, device='cuda' if self.is_cuda else 'cpu')], dim=1)

        return y_pred


class JointTrapMembershipV4(JointMembershipHyperOptimized):
    def required_dtype(self):
        return torch.float

    def left_x(self):
        return self.center - self.half_width()

    def half_width(self):
        return torch.abs(self.center_width) / 2 + torch.abs(self.side_width) + 2 / torch.abs(
            self.slope - self.slope_constraint) + self.slope_constraint

    def right_x(self):
        return self.center + self.half_width()

    def __init__(self, center, slope, center_width, side_width, constant_center=False, min_slope=0.01):
        super().__init__()
        self.slope_constraint = torch.tensor(min_slope, dtype=self.required_dtype())

        if constant_center:
            self.center = torch.tensor(center, dtype=self.required_dtype(), requires_grad=False)
        else:
            self.register_parameter('center', _mk_param(center, dtype=self.required_dtype()))

        self.register_parameter('slope', _mk_param(slope, dtype=self.required_dtype()))
        self.register_parameter('center_width', _mk_param(center_width, dtype=self.required_dtype()))
        self.register_parameter('side_width', _mk_param(side_width, dtype=self.required_dtype()))

        self.one = torch.tensor(1, dtype=self.required_dtype())
        self.n_one = torch.tensor(-1, dtype=self.required_dtype())
        self.two = torch.tensor(2, dtype=self.required_dtype())

        # FIXME Lol cheeky way to add backwards compatability
        mf_definitions = OrderedDict()
        mf_definitions["Left Edge"] = self
        mf_definitions["Left"] = self
        mf_definitions["Center"] = self
        mf_definitions["Right"] = self
        mf_definitions["Right Edge"] = self
        self.mfdefs = mf_definitions

    @property
    def num_mfs(self):
        return len(self.mfdefs)

    def fuzzify(self, x):
        output = self.forward(x)

        return (
            ('Left Edge', output[:, 0]),
            ("Left", output[:, 1]),
            ("Center", output[:, 2]),
            ("Right", output[:, 3]),
            ("Right Edge", output[:, 4])
        )

    # @profile
    def compute(self, x):
        slope = torch.abs(self.slope - self.slope_constraint) + self.slope_constraint

        slope_width = torch.reciprocal(slope)

        center_width = torch.abs(self.center_width)
        side_width = torch.abs(self.side_width)

        x = x - self.center

        center_div_2 = center_width / 2
        side_div_2 = side_width / 2

        delta1 = slope * side_div_2 + 1
        c1 = center_div_2 + slope_width + side_div_2

        c2 = center_div_2 + side_width + 2 * slope_width
        delta2 = -slope * c2 + 1

        x = torch.cat([-x, torch.abs(x + c1), torch.abs(x), torch.abs(x - c1), x], dim=1)

        x = x.T

        A = torch.diag(torch.tensor([
            slope,
            -slope,
            -slope,
            -slope,
            slope,
        ], device=x.device))

        # A = torch.sparse_coo_tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
        #                             [slope, -slope, -slope, -slope, slope], (5, 5), device=x.device)
        b = torch.tensor([
            [delta2],
            [delta1],
            [slope * center_div_2 + 1],
            [delta1],
            [delta2],
        ], device=x.device)

        # return (A @ x + b).T
        # return torch.sparse.addmm(b, A, x).T
        return torch.addmm(b, A, x).T


class JointSingleConstrainedEdgeMembershipV3(JointMembershipHyperOptimized):
    def required_dtype(self):
        return torch.float

    def left_x(self):
        return torch.abs(self.center)

    def half_width(self):
        return 1 / (2 * torch.abs(self.slope - self.slope_constraint) + self.slope_constraint)

    def right_x(self):
        return self.left_x() + self.half_width() * 2

    def __init__(self, center, slope, constant_center=False, min_slope=0.01):
        super().__init__()
        self.slope_constraint = torch.tensor(min_slope, dtype=self.required_dtype())

        if constant_center:
            self.center = torch.tensor(center, dtype=self.required_dtype(), requires_grad=False)
        else:
            self.register_parameter('center', _mk_param(center, dtype=self.required_dtype()))

        self.register_parameter('slope', _mk_param(slope, dtype=self.required_dtype()))

        # FIXME Lol cheeky way to add backwards compatability
        mf_definitions = OrderedDict()
        mf_definitions["Close"] = self
        mf_definitions["Far"] = self
        self.mfdefs = mf_definitions

    @property
    def num_mfs(self):
        return len(self.mfdefs)

    def fuzzify(self, x):
        output = self.forward(x)

        return (
            ('Close', output[:, 0]),
            ("Far", output[:, 1]),
        )

    # @profile
    def compute(self, x):
        slope = torch.abs(self.slope - self.slope_constraint) + self.slope_constraint
        center = torch.abs(self.center)

        A = torch.tensor([
            [-slope],
            [slope]
        ])

        a = center * slope
        b = torch.tensor([
            [1 + a],
            [-a]
        ])

        return torch.addmm(b, A, x.T).T


class Joint7TrapMembershipV3(JointMembershipHyperOptimized):
    def required_dtype(self):
        return torch.float

    def left_x(self):
        return self.center - self.half_width()

    def half_width(self):
        return torch.abs(self.center_width) / 2 + torch.abs(self.side_width) + torch.abs(
            self.super_side_width) + 3 / (torch.abs(self.slope - self.slope_constraint) + self.slope_constraint)

    def right_x(self):
        return self.center + self.half_width()

    def __init__(self, center, slope, center_width, side_width, super_side_width, constant_center=False,
                 min_slope=0.01):
        super().__init__()
        self.slope_constraint = torch.tensor(min_slope, dtype=self.required_dtype())

        if constant_center:
            self.center = torch.tensor(center, dtype=self.required_dtype(), requires_grad=False)
        else:
            self.register_parameter('center', _mk_param(center, dtype=self.required_dtype()))

        self.register_parameter('slope', _mk_param(slope, dtype=self.required_dtype()))
        self.register_parameter('center_width', _mk_param(center_width, dtype=self.required_dtype()))
        self.register_parameter('side_width', _mk_param(side_width, dtype=self.required_dtype()))
        self.register_parameter('super_side_width', _mk_param(super_side_width, dtype=self.required_dtype()))

        self.one = torch.tensor(1, dtype=self.required_dtype())
        self.n_one = torch.tensor(-1, dtype=self.required_dtype())
        self.two = torch.tensor(2, dtype=self.required_dtype())

        # FIXME Lol cheeky way to add backwards compatability
        mf_definitions = OrderedDict()
        mf_definitions["Left Edge"] = self
        mf_definitions["Left"] = self
        mf_definitions["Close Left"] = self
        mf_definitions["Center"] = self
        mf_definitions["Close Right"] = self
        mf_definitions["Right"] = self
        mf_definitions["Right Edge"] = self
        self.mfdefs = mf_definitions

    @property
    def num_mfs(self):
        return len(self.mfdefs)

    def fuzzify(self, x):
        output = self.forward(x)

        return (
            ('Left Edge', output[:, 0]),
            ("Left", output[:, 1]),
            ("Close Left", output[:, 2]),
            ("Center", output[:, 3]),
            ("Close Right", output[:, 4]),
            ("Right", output[:, 5]),
            ("Right Edge", output[:, 6])
        )

    # @profile
    def compute(self, x):
        # IMPLEMENT TECHNIQUE SO THAT THE OUTPUT MATRIX IS FIST ALL ZEROS.
        # When the area is 1, for the other membership functions it is 0

        slope = torch.abs(self.slope - self.slope_constraint) + self.slope_constraint

        slope_width = torch.reciprocal(slope)

        center_width = torch.abs(self.center_width)
        side_width = torch.abs(self.side_width)
        super_side_width = torch.abs(self.super_side_width)

        x = x - self.center

        center_div_2 = center_width / 2
        side_div_2 = side_width / 2
        super_side_div_2 = super_side_width / 2

        delta1 = slope * side_div_2 + 1
        c1 = center_div_2 + slope_width + side_div_2

        delta2 = slope * super_side_div_2 + 1
        c2 = center_div_2 + 2 * slope_width + side_width + super_side_div_2

        c3 = center_div_2 + side_width + 3 * slope_width + super_side_width
        delta3 = -slope * c3 + 1

        x = torch.cat([-x, torch.abs(x + c2), torch.abs(x + c1), torch.abs(x), torch.abs(x - c1), torch.abs(x - c2), x],
                      dim=1)

        x = x.T
        #
        A = torch.diag(torch.tensor([
            slope,
            -slope,
            -slope,
            -slope,
            -slope,
            -slope,
            slope,
        ], device=x.device))

        # i = [0, 1, 2, 3, 4, 5, 6]
        # A = torch.sparse_coo_tensor([i, i],
        #                             [slope, -slope, -slope, -slope, -slope, -slope, slope], (7, 7), device=x.device)
        b = torch.tensor([
            [delta3],
            [delta2],
            [delta1],
            [slope * center_div_2 + 1],
            [delta1],
            [delta2],
            [delta3],
        ], device=x.device)

        return torch.addmm(b, A, x).T
        # return torch.sparse.addmm(b, A, x).T
