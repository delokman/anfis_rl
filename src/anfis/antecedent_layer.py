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

            '''
            0 = far left
            1 = close left
            2 = zero
            3 = close right
            4 = far right
            '''
            membership_indices = [
                (1, 0),  # 1
                (1, 1),  # 2
                (1, 2),  # 3
                (1, 3),  # 4
                (1, 4),  # 5
                (3, 4),  # 6
                (3, 3),  # 7
                (3, 2),  # 8
                (3, 1),  # 9
                (3, 0),  # 10
                (1, 0),  # 11
                (1, 1),  # 12
                (1, 2),  # 13
                (1, 3),  # 14
                (1, 4),  # 15
                (2, 0),  # 16
                (2, 1),  # 17
                (2, 2),  # 18
                (2, 3),  # 19
                (2, 4),  # 20
                (3, 0),  # 21
                (3, 1),  # 22
                (3, 2),  # 23
                (3, 3),  # 24
                (3, 4),  # 25
            ]

            '''
            0 = far left
            1 = left
            2 = close left
            3 = zero
            4 = close right
            5 = right
            6 = far right
            '''
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

                (6,),  # 26
                (5,),  # 27
                (5,),  # 28
                (4,),  # 29
                (3,),  # 30
                (3,),  # 31
                (2,),  # 32
                (1,),  # 33
                (1,),  # 34
                (0,),  # 35
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


def dist_target_dist_per_theta_lookahead_theta_far_theta_near():
    """
     functions established the rule-base of the fuzzy
    :return:
    """

    '''
    controller input variables = Fuzzy Relations Control Variables (FRCVs):
    
    0 dist target
    1 dist line
    2 theta lookahead
    3 theta far
    4 theta near
    
    each tuple says which FRCVs are used for a rule with matching index i
    ie (0, 1, 3), -> (dist target, dist line, theta far) for rules 5-10.
    '''
    # FIXME for rules with uneven indices number i.e sometimes 2 or sometimes 3, pad the lesser ones with duplicates
    variable_rule_index = [
        (0, 2, 0),
        (0, 2, 0),
        (0, 2, 0),
        (0, 2, 0),
        (0, 2, 0),

        (0, 1, 3),
        (0, 1, 3),
        (0, 1, 3),
        (0, 1, 3),
        (0, 1, 3),

        (0, 1, 3),
        (0, 1, 3),
        (0, 1, 3),
        (0, 1, 3),
        (0, 1, 3),

        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),

        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),

        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),

        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),

        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
        (0, 1, 4),
    ]

    '''
    0 = near
    1 = far
    
    0 = far left
    1 = close left
    2 = zero
    3 = close right
    4 = far right
    
    0 = far left
    1 = near left
    1 = close left
    2 = zero
    3 = close right
    3 = near right
    4 = far right
    
    each FRCV in each tuple in the variable_rule_index is assigned an associated linguistic variables
    '''
    membership_indices = [
        (0, 0, 0),  # 1
        (0, 1, 0),  # 2
        (0, 2, 0),  # 3
        (0, 3, 0),  # 3
        (0, 4, 0),  # 3

        (1, 0, 0),  # 3
        (1, 0, 1),  # 3
        (1, 0, 2),  # 3
        (1, 0, 3),  # 3
        (1, 0, 4),  # 3

        (1, 6, 0),  # 3
        (1, 6, 1),  # 3
        (1, 6, 2),  # 3
        (1, 6, 3),  # 3
        (1, 6, 4),  # 3

        (1, 1, 0),  # 3
        (1, 1, 1),  # 3
        (1, 1, 2),  # 3
        (1, 1, 3),  # 3
        (1, 1, 4),  # 3

        (1, 2, 0),  # 3
        (1, 2, 1),  # 3
        (1, 2, 2),  # 3
        (1, 2, 3),  # 3
        (1, 2, 4),  # 3

        (1, 3, 0),  # 3
        (1, 3, 1),  # 3
        (1, 3, 2),  # 3
        (1, 3, 3),  # 3
        (1, 3, 4),  # 3

        (1, 4, 0),  # 3
        (1, 4, 1),  # 3
        (1, 4, 2),  # 3
        (1, 4, 3),  # 3
        (1, 4, 4),  # 3

        (1, 5, 0),  # 3
        (1, 5, 1),  # 3
        (1, 5, 2),  # 3
        (1, 5, 3),
        (1, 5, 4),
    ]

    '''
    0 = far left
    1 = left
    2 = close left
    3 = zero
    4 = close right
    5 = right
    6 = far right
    
    each rule of index i is assigned an associated angular velocity linguistic variable 
    '''
    outputs_membership = [
        (8,),  # 1
        (6,),  # 1
        (4,),  # 1
        (2,),  # 1
        (0,),  # 1

        (8,),  # 1
        (5,),  # 1
        (4,),  # 13
        (3,),  # 0
        (0,),  # 8

        (8,),  # 5
        (5,),  # 4
        (4,),  # 3
        (3,),  # 0
        (0,),  #

        (8,),  # 8
        (7,),  # 9
        (6,),  # 10
        (5,),  # 11
        (2,),  # 12

        (8,),  # 13
        (6,),  # 14
        (5,),  # 15
        (4,),  # 1
        (1,),  # 2

        (7,),  # 3
        (5,),  # 4
        (4,),  # 5
        (3,),  # 6
        (1,),  # 7

        (7,),  # 8
        (4,),  # 9
        (3,),  # 10
        (2,),  # 11
        (0,),  # 12

        (6,),  # 13
        (3,),  # 14
        (2,),  # 15
        (1,),  # 15
        (0,),  # 15
    ]

    mamdani_ruleset = {
        'variable_rule_index': variable_rule_index,
        'membership_indices': membership_indices,
        'outputs_membership': outputs_membership
    }

    return mamdani_ruleset


def dist_target_dist_per_theta_lookahead_theta_far_theta_near_with_vel():
    """
    each rule of index i is assigned an associated linear velocity linguistic variable based on the angular velocity
    linguistic variable
    """
    mamdani_ruleset = dist_target_dist_per_theta_lookahead_theta_far_theta_near()

    '''
    0 = slow
    1 = medium
    2 = fast
    '''

    output = []

    for i in mamdani_ruleset['outputs_membership']:
        x = abs(i[0] - 4)

        if x <= 1:
            o = 2
        elif x <= 3:
            o = 1
        else:
            o = 0

        output.append((o,))

    mamdani_ruleset['outputs_membership_velocity'] = output

    return mamdani_ruleset


def dist_per_theta_near_theta_far():
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

    '''
    0 = far left
    1 = close left
    2 = zero
    3 = close right
    4 = far right
    '''
    membership_indices = [
        (1, 0),  # 1
        (1, 1),  # 2
        (1, 2),  # 3
        (1, 3),  # 4
        (1, 4),  # 5
        (3, 4),  # 6
        (3, 3),  # 7
        (3, 2),  # 8
        (3, 1),  # 9
        (3, 0),  # 10
        (1, 0),  # 11
        (1, 1),  # 12
        (1, 2),  # 13
        (1, 3),  # 14
        (1, 4),  # 15
        (2, 0),  # 16
        (2, 1),  # 17
        (2, 2),  # 18
        (2, 3),  # 19
        (2, 4),  # 20
        (3, 0),  # 21
        (3, 1),  # 22
        (3, 2),  # 23
        (3, 3),  # 24
        (3, 4),  # 25
    ]

    '''
    0 = far left
    1 = left
    2 = close left
    3 = zero
    4 = close right
    5 = right
    6 = far right
    '''
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

        (6,),  # 26
        (5,),  # 27
        (5,),  # 28
        (4,),  # 29
        (3,),  # 30
        (3,),  # 31
        (2,),  # 32
        (1,),  # 33
        (1,),  # 34
        (0,),  # 35
    ]

    mamdani_ruleset = {
        'variable_rule_index': variable_rule_index,
        'membership_indices': membership_indices,
        'outputs_membership': outputs_membership
    }

    return mamdani_ruleset
