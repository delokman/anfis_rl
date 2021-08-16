import itertools

import torch


class AntecedentLayer(torch.nn.Module):
    """
        Form the 'rules' by taking all possible combinations of the MFs
        for each variable. Forward pass then calculates the fire-strengths.
    """

    def __init__(self, varlist):
        super(AntecedentLayer, self).__init__()
        # Count the (actual) mfs for each variable:
        mf_count = [var.num_mfs for var in varlist]
        # Now make the MF indices for each rule:
        mf_indices = itertools.product(*[range(n) for n in mf_count])
        self.mf_indices = torch.tensor(list(mf_indices))
        # mf_indices.shape is n_rules * n_in

    def num_rules(self):
        return len(self.mf_indices)

    def extra_repr(self, varlist=None):
        if not varlist:
            return None
        row_ants = []
        mf_count = [len(fv.mfdefs) for fv in varlist.values()]
        for rule_idx in itertools.product(*[range(n) for n in mf_count]):
            thisrule = []
            for (varname, fv), i in zip(varlist.items(), rule_idx):
                thisrule.append('{} is {}'
                                .format(varname, list(fv.mfdefs.keys())[i]))
            row_ants.append(' and '.join(thisrule))
        return '\n'.join(row_ants)

    def forward(self, x):
        """ Calculate the fire-strength for (the antecedent of) each rule
            x.shape = n_cases * n_in * n_mfs
            y.shape = n_cases * n_rules
        """
        # Expand (repeat) the rule indices to equal the batch size:
        batch_indices = self.mf_indices.expand((x.shape[0], -1, -1))
        # Then use these indices to populate the rule-antecedents
        ants = torch.gather(x.transpose(1, 2), 1, batch_indices)
        # ants.shape is n_cases * n_rules * n_in
        # Last, take the AND (= product) for each rule-antecedent
        rules = torch.prod(ants, dim=2)
        return rules


class MamdaniAntecedentLayer(torch.nn.Module):
    """
        Form the 'rules' by taking all possible combinations of the MFs
        for each variable. Forward pass then calculates the fire-strengths.
    """

    def __init__(self, mamdani_ruleset=None):
        super(MamdaniAntecedentLayer, self).__init__()
        # Count the (actual) mfs for each variable:

        if mamdani_ruleset is None:
            variable_rule_index = [
                (0, 1),  # 1
                (0, 1),  # 2
                (0, 1),  # 3
                (0, 1),  # 4
                (0, 1),  # 5
                (0, 1),  # 6
                (0, 1),  # 7
                (0, 1),  # 8
                (0, 1),  # 9
                (0, 1),  # 10
                (0, 2),  # 11
                (0, 2),  # 12
                (0, 2),  # 13
                (0, 2),  # 14
                (0, 2),  # 15
                (0, 2),  # 16
                (0, 2),  # 17
                (0, 2),  # 18
                (0, 2),  # 19
                (0, 2),  # 20
                (0, 2),  # 21
                (0, 2),  # 22
                (0, 2),  # 23
                (0, 2),  # 24
                (0, 2),  # 25
            ]

            membership_indices = [
                (0, 0),  # 1
                (0, 1),  # 2
                (0, 2),  # 3
                (0, 3),  # 4
                (0, 4),  # 5
                (1, 4),  # 6
                (1, 3),  # 7
                (1, 2),  # 8
                (1, 1),  # 9
                (1, 0),  # 10
                (2, 0),  # 11
                (2, 1),  # 12
                (2, 2),  # 13
                (2, 3),  # 14
                (2, 4),  # 15
                (3, 0),  # 16
                (3, 1),  # 17
                (3, 2),  # 18
                (3, 3),  # 19
                (3, 4),  # 20
                (4, 0),  # 21
                (4, 1),  # 22
                (4, 2),  # 23
                (4, 3),  # 24
                (4, 4),  # 25
            ]

            outputs_membership = [
                (6,),  # 1
                (5,),  # 2
                (3,),  # 3
                (1,),  # 4
                (0,),  # 5
                (0,),  # 6
                (1,),  # 7
                (3,),  # 8
                (5,),  # 9
                (6,),  # 10
                (6,),  # 11
                (5,),  # 12
                (4,),  # 13
                (3,),  # 14
                (2,),  # 15
                (5,),  # 16
                (4,),  # 17
                (3,),  # 18
                (2,),  # 19
                (1,),  # 20
                (4,),  # 21
                (3,),  # 22
                (2,),  # 23
                (1,),  # 24
                (0,),  # 25
            ]

            mamdani_ruleset = {
                'variable_rule_index': variable_rule_index,
                'membership_indices': membership_indices,
                'outputs_membership': outputs_membership
            }

        self.mamdani_ruleset = mamdani_ruleset

    def num_rules(self):
        return len(self.mamdani_ruleset['variable_rule_index'])

    def extra_repr(self, varlist=None):
        if not varlist:
            return None
        row_ants = []
        mf_count = [len(fv.mfdefs) for fv in varlist.values()]
        for rule_idx in itertools.product(*[range(n) for n in mf_count]):
            thisrule = []
            for (varname, fv), i in zip(varlist.items(), rule_idx):
                thisrule.append('{} is {}'
                                .format(varname, list(fv.mfdefs.keys())[i]))
            row_ants.append(' and '.join(thisrule))
        return '\n'.join(row_ants)

    def forward(self, x):
        """ Calculate the fire-strength for (the antecedent of) each rule
            x.shape = n_cases * n_in * n_mfs
            y.shape = n_cases * n_rules
        """

        # weights = []
        #
        # for variable, membership in zip(self.mamdani_ruleset['variable_rule_index'],
        #                                 self.mamdani_ruleset['membership_indices']):
        #     min_val, _ = torch.min(x[:, variable, membership], dim=1)
        #     weights.append(min_val)

        weights = x[:, self.mamdani_ruleset['variable_rule_index'], self.mamdani_ruleset['membership_indices']]
        # AND operation
        return torch.min(weights, dim=2)[0]
