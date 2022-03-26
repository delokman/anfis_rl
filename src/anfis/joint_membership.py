from abc import abstractmethod, ABC
from collections import OrderedDict

import numpy as np
import torch


def _mk_param(val, dtype=torch.float):
    """Make a torch parameter from a scalar value"""
    if isinstance(val, torch.Tensor):
        val = val.item()
    return torch.nn.Parameter(torch.tensor(val, dtype=dtype))


class BellMembFunc(torch.nn.Module):
    """
        Generalised Bell membership function; defined by three parameters:
            a, the half-width (at the crossover point)
            b, controls the slope at the crossover point (which is -b/2a)
            c, the center point
    """

    def __init__(self, a, b, c, constant_center=False):
        super(BellMembFunc, self).__init__()
        self.register_parameter('a', a)

        self.register_parameter('b', b)

        if constant_center:
            self.c = c
        else:
            self.register_parameter('c', c)

        self.b.register_hook(BellMembFunc.b_log_hook)

    @staticmethod
    def b_log_hook(grad):
        """
            Possibility of a log(0) in the grad for b, giving a nan.
            Fix this by replacing any nan in the grad with ~0.
        """
        grad[torch.isnan(grad)] = 1e-9
        return grad

    def forward(self, x):
        dist = torch.pow((x - self.c) / self.a, 2)
        return torch.reciprocal(1 + torch.pow(dist, self.b))

    def pretty(self):
        return 'BellMembFunc {} {} {}'.format(self.a, self.b, self.c)


class BellMembFunc2(BellMembFunc):
    """
        Generalised Bell membership function; defined by three parameters:
            a, the half-width (at the crossover point)
            b, controls the slope at the crossover point (which is -b/2a)
            c, the center point
    """

    def __init__(self, a, b, c, c2, right_side=False, constant_center=False):
        super(BellMembFunc2, self).__init__(a, b, c, constant_center=constant_center)
        self.register_parameter('c2', c2)
        self.right_side = right_side

    def forward(self, x):
        if self.right_side:
            c = self.c + self.c2
        else:
            c = self.c - self.c2

        dist = torch.pow((x - c) / self.a, 2)
        return torch.reciprocal(1 + torch.pow(dist, self.b))

    def pretty(self):
        return 'BellMembFunc {} {} {} {}'.format(self.a, self.b, self.c, self.c2)


class JointExprMembFunc(torch.nn.Module):
    """
        An inverted Exponential membership function; defined by two parameters:
            a, controls the slope at the center point (which is -1/4a)
            c, the center point
    """

    def __init__(self, a, c1, c2, c3, right_side=False, constant_center=False):
        super(JointExprMembFunc, self).__init__()
        self.register_parameter('a', a)

        if constant_center:
            self.c1 = c1
        else:
            self.register_parameter('c1', c1)

        self.register_parameter('c2', c2)
        self.register_parameter('c3', c3)
        self.right_side = right_side
        self.a.register_hook(JointExprMembFunc.a_log_hook)

    @staticmethod
    def a_log_hook(grad):
        """
            Possibility of a log(0) in the grad for b, giving a nan.
            Fix this by replacing any nan in the grad with ~0.
        """
        grad[torch.isnan(grad)] = 1e-9
        return grad

    def forward(self, x):

        if self.right_side:
            c = - x + self.c1 + self.c2 + self.c3
        else:
            c = x - self.c1 + self.c2 + self.c3
        dist = c / self.a
        return torch.reciprocal(1 + torch.exp(dist))

    def pretty(self):
        return 'ExprMembFunc {} {} {} {}'.format(self.a, self.c1, self.c2, self.c3)


class JointMembership(ABC, torch.nn.Module):
    def required_dtype(self):
        return torch.float

    def __init__(self):
        super().__init__()
        self.padding = 0
        self.is_cuda = False
        self.register_buffer("zeroes", torch.tensor([]))

    @property
    def num_mfs(self):
        """Return the actual number of MFs (ignoring any padding)"""
        return len(self.mfdefs)

    def members(self):
        """
            Return an iterator over this variables's membership functions.
            Yields tuples of the form (mf-name, MembFunc-object)
        """
        return self.mfdefs.items()

    def pad_to(self, new_size):
        """
            Will pad result of forward-pass (with zeros) so it has new_size,
            i.e. as if it had new_size MFs.
        """
        self.padding = new_size - len(self.mfdefs)

    def fuzzify(self, x):
        """
            Yield a list of (mf-name, fuzzy values) for these input values.
        """
        for mfname, mfdef in self.mfdefs.items():
            yvals = mfdef(x)
            yield mfname, yvals

    def forward(self, x):
        """
            Return a tensor giving the membership value for each MF.
            x.shape: n_cases
            y.shape: n_cases * n_mfs
        """
        y_pred = torch.cat([mf(x) for mf in self.mfdefs.values()], dim=1)
        if self.padding > 0:
            y_pred = torch.cat([y_pred,
                                torch.zeros(x.shape[0], self.padding)], dim=1)
        return y_pred

    @abstractmethod
    def left_x(self):
        pass

    @abstractmethod
    def right_x(self):
        pass


class JointBellMembership(JointMembership):

    def left_x(self):
        return self.center_1 - self.center_2 - self.center_3

    def right_x(self):
        return self.center_1 + self.center_2 + self.center_3

    def __init__(self, center_1, width_1, slope_1, center_2, width_2, slope_2, center_3, slope_3,
                 constant_center=False):
        super(JointBellMembership, self).__init__()

        if constant_center:
            self.center_1 = torch.tensor(center_1, dtype=torch.float, requires_grad=False)
        else:
            self.register_parameter('center_1', _mk_param(center_1))
        self.register_parameter('width_1', _mk_param(width_1))
        self.register_parameter('slope_1', _mk_param(slope_1))

        self.register_parameter('center_2', _mk_param(center_2))
        self.register_parameter('width_2', _mk_param(width_2))
        self.register_parameter('slope_2', _mk_param(slope_2))

        self.register_parameter('center_3', _mk_param(center_3))
        self.register_parameter('slope_3', _mk_param(slope_3))

        mf_definitions = OrderedDict()

        mf_definitions['left_edge'] = JointExprMembFunc(self.slope_3, self.center_1, self.center_2, self.center_3,
                                                        constant_center=constant_center)
        mf_definitions['left'] = BellMembFunc2(self.width_2, self.slope_2, self.center_1, self.center_2,
                                               constant_center=constant_center)
        mf_definitions['center'] = BellMembFunc(self.width_1, self.slope_1, self.center_1,
                                                constant_center=constant_center)
        mf_definitions['right'] = BellMembFunc2(self.width_2, self.slope_2, self.center_1, self.center_2,
                                                right_side=True, constant_center=constant_center)
        mf_definitions['right_edge'] = JointExprMembFunc(self.slope_3, self.center_1, self.center_2, self.center_3,
                                                         right_side=True, constant_center=constant_center)

        self.mfdefs = torch.nn.ModuleDict(mf_definitions)


class AbstractJointTrapMembFunc(ABC):
    @abstractmethod
    def shift_x(self, x):
        pass


class JointTrapMembFunc(torch.nn.Module, AbstractJointTrapMembFunc):
    def shift_x(self, x):
        return x - self.c

    def __init__(self, center, slope, width_1, min_slope, constant_center=False):
        super(JointTrapMembFunc, self).__init__()

        self.min_slope = min_slope
        if constant_center:
            self.c = center
        else:
            self.register_parameter('c', center)

        self.register_parameter('s', slope)
        self.register_parameter('w', width_1)

    def forward(self, x):
        X = torch.abs(self.shift_x(x))
        slope = torch.abs(self.s - self.min_slope) + self.min_slope

        # y_vals = torch.zeros_like(x, requires_grad=True)
        div_w = torch.div(torch.abs(self.w), 2)

        flat_area = torch.less_equal(X, div_w)
        slopped = torch.less_equal(X, div_w + torch.reciprocal(slope))
        slopped = (~flat_area) & slopped

        # one = torch.tensor(1, dtype=torch.float, requires_grad=True)
        # zero = torch.tensor(0, dtype=torch.float, requires_grad=True)
        slopped_value = - slope * (X - div_w) + 1

        # y_vals = torch.where(flat_area, one, torch.where(slopped, slopped_value, zero))

        y_vals = torch.where(flat_area, torch.ones_like(x, requires_grad=False),
                             torch.where(slopped, slopped_value,
                                         torch.zeros_like(x, requires_grad=False)))

        # y_vals[flat_area] = 1
        # y_vals[slopped] = - self.s * (X[slopped] - div_w) + 1

        return y_vals

    def pretty(self):
        return 'TrapMembFunc {} {} {}'.format(self.center, self.slope, self.width)


class JointTrapMembFunc2(JointTrapMembFunc):
    def shift_x(self, x):
        return x - self.c + self.dir * (torch.abs(self.w / 2) + torch.reciprocal(self.s) + torch.abs(self.w2) / 2)

    def __init__(self, center, slope, width_1, width_2, min_slope, constant_center=False, right_side=False):
        super(JointTrapMembFunc2, self).__init__(center, slope, width_2, min_slope, constant_center=constant_center)

        if right_side:
            self.dir = -1
        else:
            self.dir = 1

        self.register_parameter('w2', width_1)


class JointTrapEdgeMembFunc(torch.nn.Module):
    def __init__(self, center, slope, width_1, width_2, min_slope, constant_center=False, right_side=False):
        super(JointTrapEdgeMembFunc, self).__init__()

        self.min_slope = min_slope

        if constant_center:
            self.c = center
        else:
            self.register_parameter('c', center)

        if right_side:
            self.dir = 1
        else:
            self.dir = -1

        self.register_parameter('s', slope)
        self.register_parameter('w1', width_1)
        self.register_parameter('w2', width_2)

    def forward(self, x):

        slope = torch.abs(self.s - self.min_slope) + self.min_slope
        slope_w = torch.reciprocal(slope)

        X = self.dir * (x - self.c) - (torch.abs(self.w1) / 2 + torch.abs(self.w2) + 2 * slope_w)

        # y_vals = torch.zeros_like(x, requires_grad=True)

        flat_area = torch.greater(X, 0)
        slopped = torch.greater_equal(X, -slope_w)
        slopped = (~flat_area) & slopped

        # one = torch.tensor(1, dtype=torch.float,requires_grad=True)
        # zero = torch.tensor(0, dtype=torch.float,requires_grad=True)
        slopped_value = slope * (X + slope_w)

        y_vals = torch.where(flat_area, torch.ones_like(x, requires_grad=False),
                             torch.where(slopped, slopped_value,
                                         torch.zeros_like(x, requires_grad=False)))

        # from torch.nn.functional import relu
        # relu()
        # y_vals[flat_area] = 1
        # y_vals[slopped] = self.s * (X[slopped] + slope_w)

        return y_vals

    def pretty(self):
        return 'TrapMembFunc {} {} {}'.format(self.center, self.slope, self.width)


class JointTrapMembership(JointMembership):

    def left_x(self):
        return self.center - self.half_width()

    def half_width(self):
        return torch.abs(self.width_1) / 2 + torch.abs(self.width_2) + 2 / torch.abs(
            self.slope - self.min_slope) + self.min_slope

    def right_x(self):
        return self.center + self.half_width()

    def __init__(self, center, slope, width_1, width_2,
                 constant_center=False, min_slope=0.01):
        super(JointTrapMembership, self).__init__()
        self.min_slope = torch.tensor(min_slope, dtype=torch.float)

        if constant_center:
            self.center = torch.tensor(center, dtype=torch.float, requires_grad=False)
        else:
            self.register_parameter('center', _mk_param(center))

        self.register_parameter('slope', _mk_param(slope))
        self.register_parameter('width_1', _mk_param(width_1))
        self.register_parameter('width_2', _mk_param(width_2))

        mf_definitions = OrderedDict()

        mf_definitions['left_edge'] = JointTrapEdgeMembFunc(self.center, self.slope, self.width_1, self.width_2,
                                                            self.min_slope, constant_center=constant_center)
        mf_definitions['left'] = JointTrapMembFunc2(self.center, self.slope, self.width_1, self.width_2,
                                                    self.min_slope, constant_center=constant_center)
        mf_definitions['center'] = JointTrapMembFunc(self.center, self.slope, self.width_1,
                                                     self.min_slope, constant_center=constant_center)
        mf_definitions['right'] = JointTrapMembFunc2(self.center, self.slope, self.width_1, self.width_2,
                                                     self.min_slope, right_side=True, constant_center=constant_center)
        mf_definitions['right_edge'] = JointTrapEdgeMembFunc(self.center, self.slope, self.width_1, self.width_2,
                                                             self.min_slope, right_side=True,
                                                             constant_center=constant_center)

        self.mfdefs = torch.nn.ModuleDict(mf_definitions)


class AbstractJointDiffSigmoidMembFunc(ABC):
    @abstractmethod
    def shift_x(self, x, m_s):
        pass

    @abstractmethod
    def get_width(self, ms):
        pass


class JointDiffSigmoidMembFunc(torch.nn.Module, AbstractJointDiffSigmoidMembFunc):
    def shift_x(self, x, m_s):
        return x

    def get_width(self, ms):
        return self.get_center_width(ms)

    def get_center_width(self, m_s):
        return torch.abs(self.center_width - m_s) + m_s

    def __init__(self, center, slope, center_width, width_constrain, min_slope, constant_center=False):
        super(JointDiffSigmoidMembFunc, self).__init__()

        self.min_slope = min_slope
        self.width_constrain = width_constrain
        if constant_center:
            self.c = center
        else:
            self.register_parameter('c', center)

        self.register_parameter('s', slope)
        self.register_parameter('center_width', center_width)

    def forward(self, x):
        s = torch.abs(self.s - self.min_slope) + self.min_slope
        m_s = torch.div(self.width_constrain, s)

        w = self.get_width(m_s)

        x = self.shift_x(x - self.c, m_s)

        # Need to convert to 64 bit float otherwise value is inf
        data = torch.cosh(s * x.double())

        return torch.div(torch.sinh(s * w / 2), torch.add(torch.cosh(s * w / 2), data)).float()

    def pretty(self):
        return 'SigmoidMembFunc {} {} {}'.format(self.center, self.slope, self.center_width)


class JointDiffSigmoidMembFunc2(JointDiffSigmoidMembFunc):
    def shift_x(self, x, m_s):
        return x + self.dir * (self.get_center_width(m_s) / 2 + self.get_side_width(m_s) / 2)

    def get_width(self, ms):
        return self.get_side_width(ms)

    def get_side_width(self, m_s):
        return torch.abs(self.side_width - m_s) + m_s

    def __init__(self, center, slope, center_width, side_width, width_constrain, min_slope, constant_center=False,
                 right_side=False):
        super(JointDiffSigmoidMembFunc2, self).__init__(center, slope, center_width, width_constrain, min_slope,
                                                        constant_center=constant_center)

        if right_side:
            self.dir = -1
        else:
            self.dir = 1

        self.register_parameter('side_width', side_width)

    def pretty(self):
        return 'SigmoidMembFunc {} {} {} {}'.format(self.center, self.slope, self.center_width, self.side_width)


class JointDiffSigmoidEdgeMembFunc(torch.nn.Module):

    def __init__(self, center, slope, center_width,
                 side_width, width_constrain, min_slope,
                 constant_center=True, right_side=False):
        super().__init__()

        self.min_slope = min_slope
        self.width_constrain = width_constrain
        if constant_center:
            self.c = center
        else:
            self.register_parameter('c', center)

        if right_side:
            self.dir = 1
        else:
            self.dir = -1

        self.register_parameter('s', slope)
        self.register_parameter('center_width', center_width)
        self.register_parameter('side_width', side_width)

    def forward(self, x):
        s = torch.abs(self.s - self.min_slope) + self.min_slope
        m_s = torch.div(self.width_constrain, s)

        center_width = torch.abs(self.center_width - m_s) + m_s
        side_width = torch.abs(self.side_width - m_s) + m_s

        # Fixed overflow by setting x to be a float64
        exp = torch.exp(-s * (self.dir * (x.double() - self.c) - (center_width / 2 + side_width)))

        return torch.reciprocal(1 + exp).float()

    def pretty(self):
        return 'SigmoidMembFunc {} {} {}'.format(self.center, self.slope, self.center_width)


class JointDiffSigmoidMembership(JointMembership):
    def left_x(self):
        return self.center - self.half_width()

    def half_width(self):
        slope = torch.abs(self.slope - self.slope_constraint) + self.slope_constraint

        ms = self.width_constraint / slope

        return (torch.abs(self.center_width - ms) + ms) / 2 + torch.abs(self.side_width - ms) + ms + 8 / slope

    def right_x(self):
        return self.center + self.half_width()

    def __init__(self, center, slope, center_width, side_width, min_threshold=0.99, min_slope=0.1,
                 constant_center=True):
        super().__init__()
        self.width_constraint = torch.tensor(np.arctanh(min_threshold) * 4, dtype=torch.float)
        self.slope_constraint = torch.tensor(min_slope, dtype=torch.float)

        if constant_center:
            self.center = torch.tensor(center, dtype=torch.float, requires_grad=False)
        else:
            self.register_parameter('center', _mk_param(center))

        self.register_parameter('slope', _mk_param(slope))
        self.register_parameter('center_width', _mk_param(center_width))
        self.register_parameter('side_width', _mk_param(side_width))

        mf_definitions = OrderedDict()

        mf_definitions['left_edge'] = JointDiffSigmoidEdgeMembFunc(self.center, self.slope, self.center_width,
                                                                   self.side_width, self.width_constraint,
                                                                   self.slope_constraint,
                                                                   constant_center=constant_center)
        mf_definitions['left'] = JointDiffSigmoidMembFunc2(self.center, self.slope, self.center_width, self.side_width,
                                                           self.width_constraint, self.slope_constraint,
                                                           constant_center=constant_center)
        mf_definitions['center'] = JointDiffSigmoidMembFunc(self.center, self.slope, self.center_width,
                                                            self.width_constraint, self.slope_constraint,
                                                            constant_center=constant_center)
        mf_definitions['right'] = JointDiffSigmoidMembFunc2(self.center, self.slope, self.center_width, self.side_width,
                                                            self.width_constraint, self.slope_constraint,
                                                            right_side=True, constant_center=constant_center)
        mf_definitions['right_edge'] = JointDiffSigmoidEdgeMembFunc(self.center, self.slope, self.center_width,
                                                                    self.side_width, self.width_constraint,
                                                                    self.slope_constraint,
                                                                    right_side=True, constant_center=constant_center)

        self.mfdefs = torch.nn.ModuleDict(mf_definitions)


def get_quantiles(dataset):
    x = dataset.dataset.tensors[0]

    q1 = torch.quantile(x, .25, dim=0)
    q2 = torch.quantile(x, .75, dim=0)

    IQR = q2 - q1

    return IQR


def best_initial_diff_sigmoid_parameter(dataset, tolerance=0.99):
    x_range = 2 * get_quantiles(dataset)

    t = np.arctanh(tolerance) * 4

    s = (3 * t + 16) / (2 * x_range)
    w = t / s

    zero = torch.zeros_like(s)

    return torch.stack([zero, s, w, w]).t()


def best_initial_trapezoid_parameter(dataset):
    x_range = 2 * get_quantiles(dataset)

    # s = 2 / x_range
    # zero = torch.zeros_like(s)
    # return torch.stack([zero, s, zero, zero]).t()

    w = 2 * x_range / 7
    s = 1 / w
    zero = torch.zeros_like(s)

    return torch.stack([zero, s, w, w]).t()
