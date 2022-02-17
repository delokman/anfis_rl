import torch

from anfis.joint_membership import _mk_param


class Matrix5ErrorJointFuzzifyLayer(torch.nn.Module):
    """
        A list of fuzzy variables, representing the inputs to the FIS.
        Forward pass will fuzzify each variable individually.
        We pad the variables so they all seem to have the same number of MFs,
        as this allows us to put all results in the same tensor.
    """

    def required_dtype(self):
        return torch.float

    def __init__(self, td_c, td_s,
                 dl_c, dl_s, dl_cw, dl_sw, dl_ssw,
                 tl_c, tl_s, tl_cw, tl_sw,
                 tf_c, tf_s, tf_cw, tf_sw,
                 tn_c, tn_s, tn_cw, tn_sw,
                 td_constant_c=True,
                 dl_constant_c=False,
                 tl_constant_c=False,
                 tf_constant_c=False,
                 tn_constant_c=False,
                 min_slope=0.01):
        super(Matrix5ErrorJointFuzzifyLayer, self).__init__()
        self.is_cuda = False

        self.slope_constraint = torch.tensor(min_slope, dtype=self.required_dtype())

        # Target distance error
        if td_constant_c:
            self.td_c = torch.tensor(td_c, dtype=self.required_dtype(), requires_grad=False)
        else:
            self.register_parameter('td_c', _mk_param(td_c, dtype=self.required_dtype()))
        self.register_parameter('td_s', _mk_param(td_s, dtype=self.required_dtype()))

        # Distance line error
        if dl_constant_c:
            self.dl_c = torch.tensor(dl_c, dtype=self.required_dtype(), requires_grad=False)
        else:
            self.register_parameter('dl_c', _mk_param(dl_c, dtype=self.required_dtype()))
        self.register_parameter('dl_s', _mk_param(dl_s, dtype=self.required_dtype()))
        self.register_parameter('dl_cw', _mk_param(dl_cw, dtype=self.required_dtype()))
        self.register_parameter('dl_sw', _mk_param(dl_sw, dtype=self.required_dtype()))
        self.register_parameter('dl_ssw', _mk_param(dl_ssw, dtype=self.required_dtype()))

        # Theta lookahead
        if tl_constant_c:
            self.tl_c = torch.tensor(tl_c, dtype=self.required_dtype(), requires_grad=False)
        else:
            self.register_parameter('tl_c', _mk_param(tl_c, dtype=self.required_dtype()))
        self.register_parameter('tl_s', _mk_param(tl_s, dtype=self.required_dtype()))
        self.register_parameter('tl_cw', _mk_param(tl_cw, dtype=self.required_dtype()))
        self.register_parameter('tl_sw', _mk_param(tl_sw, dtype=self.required_dtype()))

        # Theta far
        if tf_constant_c:
            self.tf_c = torch.tensor(tf_c, dtype=self.required_dtype(), requires_grad=False)
        else:
            self.register_parameter('tf_c', _mk_param(tf_c, dtype=self.required_dtype()))
        self.register_parameter('tf_s', _mk_param(tf_s, dtype=self.required_dtype()))
        self.register_parameter('tf_cw', _mk_param(tf_cw, dtype=self.required_dtype()))
        self.register_parameter('tf_sw', _mk_param(tf_sw, dtype=self.required_dtype()))

        # Theta near
        if tn_constant_c:
            self.tn_c = torch.tensor(tn_c, dtype=self.required_dtype(), requires_grad=False)
        else:
            self.register_parameter('tn_c', _mk_param(tn_c, dtype=self.required_dtype()))
        self.register_parameter('tn_s', _mk_param(tn_s, dtype=self.required_dtype()))
        self.register_parameter('tn_cw', _mk_param(tn_cw, dtype=self.required_dtype()))
        self.register_parameter('tn_sw', _mk_param(tn_sw, dtype=self.required_dtype()))

    @property
    def num_in(self):
        """Return the number of input variables"""
        return 5

    @property
    def max_mfs(self):
        """ Return the max number of MFs in any variable"""
        return 7

    def forward(self, x):
        """ Fuzzyify each variable's value using each of its corresponding mfs.
            x.shape = n_cases * n_in
            y.shape = n_cases * n_in * n_mfs
        """
        if self.is_cuda and not x.is_cuda:
            x = x.cuda()

        # SLOPE CONSTANT
        td_s = torch.abs(self.td_s - self.slope_constraint) + self.slope_constraint
        dl_s = torch.abs(self.dl_s - self.slope_constraint) + self.slope_constraint
        tl_S = torch.abs(self.tl_s - self.slope_constraint) + self.slope_constraint
        tf_s = torch.abs(self.tf_s - self.slope_constraint) + self.slope_constraint
        tn_s = torch.abs(self.tn_s - self.slope_constraint) + self.slope_constraint

        A = torch.diag(torch.tensor([
            # Target distance
            -td_s,
            td_s,

            # Target line
            dl_s,
            -dl_s,
            -dl_s,
            -dl_s,
            -dl_s,
            -dl_s,
            dl_s,

        ]))

        # Distance line
        dl_slope_width = torch.reciprocal(dl_s)

        dl_center_width = torch.abs(self.dl_cw)
        dl_side_width = torch.abs(self.dl_sw)
        dl_super_side_width = torch.abs(self.dl_ssw)

        dl_center_div_2 = dl_center_width / 2
        dl_side_div_2 = dl_side_width / 2
        dl_super_side_div_2 = dl_super_side_width / 2

        dl_delta1 = dl_s * dl_side_div_2 + 1
        dl_c1 = dl_center_div_2 + dl_slope_width + dl_side_div_2

        dl_delta2 = dl_s * dl_super_side_div_2 + 1
        dl_c2 = dl_center_div_2 + 2 * dl_slope_width + dl_side_width + dl_super_side_div_2

        dl_c3 = dl_center_div_2 + dl_side_width + 3 * dl_slope_width + dl_super_side_width
        dl_delta3 = -dl_s * dl_c3 + 1

        # Theta lookahead
        # Theta far
        # Theta near

        a = self.td_c * td_s
        b = torch.tensor([
            # Target distance
            [1 + a],
            [-a],

            #     Target Line
            [dl_delta3],
            [dl_delta2],
            [dl_delta1],
            [dl_s * dl_center_div_2 + 1],
            [dl_delta1],
            [dl_delta2],
            [dl_delta3],

        ])

        x = x.unsqueeze(1)
        aug_x = torch.cat([x, x,
                           -x, torch.abs(x + dl_c2), torch.abs(x + dl_c1), torch.abs(x), torch.abs(x - dl_c1),
                           torch.abs(x - dl_c2), x

                           ], dim=1)

        y_pred = torch.addmm(b, A, aug_x.T)

        return torch.clamp_(y_pred, 0, 1)
