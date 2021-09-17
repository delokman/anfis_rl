from collections import OrderedDict

import numpy as np
import torch

from anfis.joint_membership import _mk_param, JointMembership


class JointDiffSigmoidMembershipV2(JointMembership):
    def required_dtype(self):
        return torch.float64

    def left_x(self):
        return self.center - self.half_width()

    def half_width(self):
        with torch.no_grad():
            slope = torch.abs(self.slope - self.slope_constraint) + self.slope_constraint
            ms = self.width_constraint / slope

            center_width = torch.abs(self.center_width - ms) + ms
            side_width = torch.abs(self.side_width - ms) + ms

            return center_width / 2 + side_width + 8 / slope

    def right_x(self):
        return self.center + self.half_width()

    def __init__(self, center, slope, center_width, side_width, min_threshold=0.99, min_slope=0.1,
                 constant_center=True):
        super().__init__()
        self.width_constraint = torch.tensor(np.arctanh(min_threshold) * 4, dtype=torch.float64)
        self.slope_constraint = torch.tensor(min_slope, dtype=torch.float64)

        if constant_center:
            self.center = torch.tensor(center, dtype=torch.float64, requires_grad=False)
        else:
            self.register_parameter('center', _mk_param(center, dtype=torch.float64))

        self.register_parameter('slope', _mk_param(slope, dtype=torch.float64))
        self.register_parameter('center_width', _mk_param(center_width, dtype=torch.float64))
        self.register_parameter('side_width', _mk_param(side_width, dtype=torch.float64))

        self.one = torch.tensor(1, dtype=torch.float64)
        self.n_one = torch.tensor(-1, dtype=torch.float64)
        self.two = torch.tensor(2, dtype=torch.float64)

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

    def forward(self, x):
        slope = torch.abs(self.slope - self.slope_constraint) + self.slope_constraint
        ms = self.width_constraint / slope

        center_width = torch.abs(self.center_width - ms) + ms
        side_width = torch.abs(self.side_width - ms) + ms

        x = x - self.center

        f1 = slope * center_width / self.two
        f2 = slope * side_width / self.two

        sinh2 = torch.sinh(f2)
        cosh2 = torch.cosh(f2)

        center = torch.true_divide(torch.sinh(f1), torch.add(torch.cosh(f1), torch.cosh(slope * x)))
        shift = (center_width / self.two + side_width / self.two) * slope
        left = torch.true_divide(sinh2, torch.add(cosh2, torch.cosh(slope * x + shift)))
        right = torch.true_divide(sinh2, torch.add(cosh2, torch.cosh(slope * x - shift)))

        shift = (center_width / self.two + side_width) * slope
        left_edge = torch.reciprocal(self.one + torch.exp(slope * x + shift))
        right_edge = torch.reciprocal(self.one + torch.exp(-slope * x + shift))

        return torch.cat([left_edge, left, center, right, right_edge], dim=1)


class JointTrapMembershipV2(JointMembership):
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

    def forward(self, x):
        # IMPLEMENT TECHNIQUE SO THAT THE OUTPUT MATRIX IS FIST ALL ZEROS.
        # When the area is 1, for the other membership functions it is 0

        slope = torch.abs(self.slope - self.slope_constraint) + self.slope_constraint

        slope_width = torch.reciprocal(slope)

        center_width = torch.abs(self.center_width)
        side_width = torch.abs(self.side_width)

        x = x - self.center

        ones = torch.ones_like(x, requires_grad=False)
        zeros = torch.zeros_like(x, requires_grad=False)

        # CENTER

        x_center = torch.abs(x)

        center_div_2 = torch.div(center_width, 2)

        flat_area = torch.less_equal(x_center, center_div_2)
        slopped = torch.less_equal(x_center, center_div_2 + slope_width)
        slopped = (~flat_area) & slopped
        slopped_value = - slope * (x_center - center_div_2) + 1

        center = torch.where(flat_area, ones, torch.where(slopped, slopped_value, zeros))

        # LEFT AND RIGHT

        side_div_2 = torch.div(side_width, 2)
        shift = center_div_2 + side_div_2 + slope_width

        mul_slope = slope * side_div_2 + 1

        # LEFT
        x_left = torch.abs(x + shift)

        flat_area = torch.less_equal(x_left, side_div_2)
        slopped = torch.less_equal(x_left, side_div_2 + slope_width)
        slopped = (~flat_area) & slopped
        # slopped_value = - slope * (x_left - side_div_2) + 1
        slopped_value = - slope * x_left + mul_slope
        left = torch.where(flat_area, ones, torch.where(slopped, slopped_value, zeros))

        # RIGHT
        x_right = torch.abs(x - shift)

        flat_area = torch.less_equal(x_right, side_div_2)
        slopped = torch.less_equal(x_right, side_div_2 + slope_width)
        slopped = ~flat_area & slopped
        # slopped_value = - slope * (x_right - side_div_2) + 1
        slopped_value = - slope * x_right + mul_slope
        right = torch.where(flat_area, ones, torch.where(slopped, slopped_value, zeros))

        # EDGES

        shift = (center_div_2 + side_width + 2 * slope_width)

        mul_slope = slope * slope_width

        # LEFT EDGE
        x_left_edge = -x - shift

        flat_area = torch.greater(x_left_edge, 0)
        slopped = torch.greater_equal(x_left_edge, -slope_width)
        slopped = (~flat_area) & slopped
        slopped_value = slope * x_left_edge + mul_slope

        left_edge = torch.where(flat_area, ones, torch.where(slopped, slopped_value, zeros))

        # RIGHT EDGE
        x_right_edge = x - shift

        flat_area = torch.greater(x_right_edge, 0)
        slopped = torch.greater_equal(x_right_edge, -slope_width)
        slopped = (~flat_area) & slopped
        slopped_value = slope * x_right_edge + mul_slope

        right_edge = torch.where(flat_area, ones, torch.where(slopped, slopped_value, zeros))

        return torch.cat([left_edge, left, center, right, right_edge], dim=1)


class JointSingleConstrainedEdgeMembership(JointMembership):
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

    def forward(self, x):
        slope = torch.abs(self.slope - self.slope_constraint) + self.slope_constraint
        center = torch.abs(self.center)

        slope_width = torch.reciprocal(slope)

        x = x - center

        ones = torch.ones_like(x, requires_grad=False)
        zeros = torch.zeros_like(x, requires_grad=False)

        flat_area = torch.greater(x, slope_width)
        slopped = torch.greater_equal(x, 0)
        slopped = (~flat_area) & slopped
        slopped_value = slope * x

        right = torch.where(flat_area, ones, torch.where(slopped, slopped_value, zeros))
        left = 1 - right

        return torch.cat([left, right], dim=1)


class Joint7TrapMembership(JointMembership):
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

    def forward(self, x):
        # IMPLEMENT TECHNIQUE SO THAT THE OUTPUT MATRIX IS FIST ALL ZEROS.
        # When the area is 1, for the other membership functions it is 0

        slope = torch.abs(self.slope - self.slope_constraint) + self.slope_constraint

        slope_width = torch.reciprocal(slope)

        center_width = torch.abs(self.center_width)
        side_width = torch.abs(self.side_width)
        super_side_width = torch.abs(self.super_side_width)

        x = x - self.center

        ones = torch.ones_like(x, requires_grad=False)
        zeros = torch.zeros_like(x, requires_grad=False)

        # CENTER

        x_center = torch.abs(x)

        center_div_2 = torch.div(center_width, 2)

        flat_area = torch.less_equal(x_center, center_div_2)
        slopped = torch.less_equal(x_center, center_div_2 + slope_width)
        slopped = (~flat_area) & slopped
        slopped_value = - slope * (x_center - center_div_2) + 1

        center = torch.where(flat_area, ones, torch.where(slopped, slopped_value, zeros))

        # CLOSE LEFT AND RIGHT

        side_div_2 = torch.div(side_width, 2)
        shift = center_div_2 + side_div_2 + slope_width

        mul_slope = slope * side_div_2 + 1

        # LEFT
        x_left = torch.abs(x + shift)

        flat_area = torch.less_equal(x_left, side_div_2)
        slopped = torch.less_equal(x_left, side_div_2 + slope_width)
        slopped = (~flat_area) & slopped
        # slopped_value = - slope * (x_left - side_div_2) + 1
        slopped_value = - slope * x_left + mul_slope
        close_left = torch.where(flat_area, ones, torch.where(slopped, slopped_value, zeros))

        # RIGHT
        x_right = torch.abs(x - shift)

        flat_area = torch.less_equal(x_right, side_div_2)
        slopped = torch.less_equal(x_right, side_div_2 + slope_width)
        slopped = ~flat_area & slopped
        # slopped_value = - slope * (x_right - side_div_2) + 1
        slopped_value = - slope * x_right + mul_slope
        close_right = torch.where(flat_area, ones, torch.where(slopped, slopped_value, zeros))

        # LEFT AND RIGHT

        super_side_div_2 = torch.div(super_side_width, 2)
        shift = center_div_2 + side_width + super_side_div_2 + 2 * slope_width
        mul_slope = slope * super_side_div_2 + 1

        # LEFT
        x_left = torch.abs(x + shift)

        flat_area = torch.less_equal(x_left, super_side_div_2)
        slopped = torch.less_equal(x_left, super_side_div_2 + slope_width)
        slopped = (~flat_area) & slopped
        # slopped_value = - slope * (x_left - side_div_2) + 1
        slopped_value = - slope * x_left + mul_slope
        left = torch.where(flat_area, ones, torch.where(slopped, slopped_value, zeros))

        # RIGHT
        x_right = torch.abs(x - shift)

        flat_area = torch.less_equal(x_right, super_side_div_2)
        slopped = torch.less_equal(x_right, super_side_div_2 + slope_width)
        slopped = ~flat_area & slopped
        # slopped_value = - slope * (x_right - side_div_2) + 1
        slopped_value = - slope * x_right + mul_slope
        right = torch.where(flat_area, ones, torch.where(slopped, slopped_value, zeros))

        # EDGES

        shift = (center_div_2 + side_width + 3 * slope_width + super_side_width)

        mul_slope = slope * slope_width

        # LEFT EDGE
        x_left_edge = -x - shift

        flat_area = torch.greater(x_left_edge, 0)
        slopped = torch.greater_equal(x_left_edge, -slope_width)
        slopped = (~flat_area) & slopped
        slopped_value = slope * x_left_edge + mul_slope

        left_edge = torch.where(flat_area, ones, torch.where(slopped, slopped_value, zeros))

        # RIGHT EDGE
        x_right_edge = x - shift

        flat_area = torch.greater(x_right_edge, 0)
        slopped = torch.greater_equal(x_right_edge, -slope_width)
        slopped = (~flat_area) & slopped
        slopped_value = slope * x_right_edge + mul_slope

        right_edge = torch.where(flat_area, ones, torch.where(slopped, slopped_value, zeros))

        return torch.cat([left_edge, left, close_left, center, close_right, right, right_edge], dim=1)
