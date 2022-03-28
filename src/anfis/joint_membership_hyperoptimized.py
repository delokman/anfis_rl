from abc import abstractmethod
from collections import OrderedDict

import torch

from anfis.joint_membership import _mk_param, JointMembership


# from line_profiler_pycharm import profile


class JointMembershipHyperOptimized(JointMembership):
    def __init__(self):
        super(JointMembershipHyperOptimized, self).__init__()

    @abstractmethod
    def compute(self, x):
        pass

    # @profile
    def forward(self, x):
        if self.zeroes.is_cuda and not x.is_cuda:
            x = x.cuda()

        y_pred = self.compute(x)
        torch.clamp_(y_pred, 0, 1)
        # torch.clamp_max_(y_pred, 1)
        # mask = y_pred < 0
        # a = 0.01
        # y_pred[mask] = a * (torch.exp(y_pred[mask]) - 1)

        # a = 1.6732632423543772848170429916717
        # l = 1.0507009873554804934193349852946
        # y_pred[mask] = a * torch.exp(y_pred[mask]) - a
        # y_pred *= l

        # scale = .05
        #
        # y_pred = scale * torch.nn.SELU()(y_pred / scale)
        #

        # a = 10
        # y_pred = 0.5 * y_pred * (1 + torch.tanh(a * np.sqrt(2 / np.pi) * (y_pred + torch.pow(y_pred, 3))))
        # torch.clamp_max_(y_pred, 1)

        if self.padding > 0:
            self.zeroes = torch.zeros(x.shape[0], self.padding, device=y_pred.device)
            y_pred = torch.cat([y_pred, self.zeroes], dim=1)

        return y_pred


class JointTrapMembershipV3(JointMembershipHyperOptimized):
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
        self.register_buffer("slope_constraint",  torch.tensor(min_slope, dtype=self.required_dtype()))

        if constant_center:
            self.register_buffer("center", torch.tensor(center, dtype=self.required_dtype(), requires_grad=False))
        else:
            self.register_parameter('center', _mk_param(center, dtype=self.required_dtype()))

        self.register_parameter('slope', _mk_param(slope, dtype=self.required_dtype()))
        self.register_parameter('center_width', _mk_param(center_width, dtype=self.required_dtype()))
        self.register_parameter('side_width', _mk_param(side_width, dtype=self.required_dtype()))

        self.register_buffer("one", torch.tensor(1, dtype=self.required_dtype()))
        self.register_buffer("n_one",  torch.tensor(-1, dtype=self.required_dtype()))
        self.register_buffer("two", torch.tensor(2, dtype=self.required_dtype()))

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

        # CENTER

        x_center = torch.abs(x)
        center = - slope * (x_center - center_div_2) + 1

        # CLOSE LEFT AND RIGHT

        shift = slope * side_div_2 + 1
        c = center_div_2 + slope_width + side_div_2

        # LEFT

        x_close_left = torch.abs(x + c)
        close_left = -slope * x_close_left + shift

        # RIGHT
        x_close_right = torch.abs(x - c)
        close_right = -slope * x_close_right + shift

        # EDGES

        c = center_div_2 + side_width + 2 * slope_width

        # LEFT EDGE
        x_left_edge = -x - c
        left_edge = slope * x_left_edge + 1

        # RIGHT EDGE
        x_right_edge = x - c
        right_edge = slope * x_right_edge + 1

        return torch.cat([left_edge, close_left, center, close_right, right_edge], dim=1)


class JointSingleConstrainedEdgeMembershipV2(JointMembershipHyperOptimized):
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
        self.register_buffer("slope_constraint",  torch.tensor(min_slope, dtype=self.required_dtype()))

        if constant_center:
            self.register_buffer("center", torch.tensor(center, dtype=self.required_dtype(), requires_grad=False))
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

        x = x - center

        right = slope * x
        left = 1 - right

        return torch.cat([left, right], dim=1)


class Joint7TrapMembershipV2(JointMembershipHyperOptimized):
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
        self.register_buffer("slope_constraint",  torch.tensor(min_slope, dtype=self.required_dtype()))

        if constant_center:
            self.register_buffer("center", torch.tensor(center, dtype=self.required_dtype(), requires_grad=False))
        else:
            self.register_parameter('center', _mk_param(center, dtype=self.required_dtype()))

        self.register_parameter('slope', _mk_param(slope, dtype=self.required_dtype()))
        self.register_parameter('center_width', _mk_param(center_width, dtype=self.required_dtype()))
        self.register_parameter('side_width', _mk_param(side_width, dtype=self.required_dtype()))
        self.register_parameter('super_side_width', _mk_param(super_side_width, dtype=self.required_dtype()))

        self.register_buffer("one", torch.tensor(1, dtype=self.required_dtype()))
        self.register_buffer("n_one",  torch.tensor(-1, dtype=self.required_dtype()))
        self.register_buffer("two", torch.tensor(2, dtype=self.required_dtype()))

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

        # CENTER

        x_center = torch.abs(x)
        center = - slope * (x_center - center_div_2) + 1

        # CLOSE LEFT AND RIGHT

        shift = slope * side_div_2 + 1
        c = center_div_2 + slope_width + side_div_2

        # LEFT

        x_close_left = torch.abs(x + c)
        close_left = -slope * x_close_left + shift

        # RIGHT
        x_close_right = torch.abs(x - c)
        close_right = -slope * x_close_right + shift

        # LEFT AND RIGHT

        shift = slope * super_side_div_2 + 1
        c = center_div_2 + 2 * slope_width + side_width + super_side_div_2

        # LEFT

        x_left = torch.abs(x + c)
        left = -slope * x_left + shift

        # RIGHT

        x_right = torch.abs(x - c)
        right = -slope * x_right + shift

        # EDGES

        c = center_div_2 + side_width + 3 * slope_width + super_side_width

        # LEFT EDGE
        x_left_edge = -x - c
        left_edge = slope * x_left_edge + 1

        # RIGHT EDGE
        x_right_edge = x - c
        right_edge = slope * x_right_edge + 1

        return torch.cat([left_edge, left, close_left, center, close_right, right, right_edge], dim=1)
