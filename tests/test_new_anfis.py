import unittest

import numpy as np
import torch

from new_ddpg.input_membership import JointTrapMembership
from new_anfis import JointAnfisNet
from new_ddpg.output_membership import SymmetricCenterOfMaximum, CenterOfMaximum
from rl.predifined_anfis import dist_target_dist_per_theta_lookahead_theta_far_theta_near_with_vel


class TestNewANFIS(unittest.TestCase):
    def setUp(self) -> None:
        mem1 = JointTrapMembership(0, np.log(np.array([1.])))
        mem2 = JointTrapMembership(0, np.log(np.array([1., 1., 1., 1, 1, 1])))
        mem3 = JointTrapMembership(0, np.log(np.array([1., 1., 1, 1])))
        mem4 = JointTrapMembership(0, np.log(np.array([1., 1., 1, 1])))
        mem5 = JointTrapMembership(0, np.log(np.array([1., 1., 1, 1])))

        ruleset = dist_target_dist_per_theta_lookahead_theta_far_theta_near_with_vel()

        out1 = SymmetricCenterOfMaximum(0., [1., 1., 1., 1.])
        out2 = CenterOfMaximum([0, 1., 1])

        self.anfis = JointAnfisNet([mem1, mem2, mem3, mem4, mem5], [out1, out2], ruleset, [4, 2], [-4, 0])
        self.anfis.set_training_mode(False)

    def test_input_rules(self):
        input_rules = self.anfis.input_rules

        true_rulebase = torch.tensor(
            [[0, 9, 0], [0, 10, 0], [0, 11, 0], [0, 12, 0], [0, 13, 0], [1, 2, 14], [1, 2, 15], [1, 2, 16], [1, 2, 17],
             [1, 2, 18], [1, 8, 14], [1, 8, 15], [1, 8, 16], [1, 8, 17], [1, 8, 18], [1, 3, 19], [1, 3, 20], [1, 3, 21],
             [1, 3, 22], [1, 3, 23], [1, 4, 19], [1, 4, 20], [1, 4, 21], [1, 4, 22], [1, 4, 23], [1, 5, 19], [1, 5, 20],
             [1, 5, 21], [1, 5, 22], [1, 5, 23], [1, 6, 19], [1, 6, 20], [1, 6, 21], [1, 6, 22], [1, 6, 23], [1, 7, 19],
             [1, 7, 20], [1, 7, 21], [1, 7, 22], [1, 7, 23]])

        self.assertTrue(torch.equal(input_rules, true_rulebase))

    def test_output_rules(self):
        input_rules = self.anfis.output_rules

        true_rulebase = torch.tensor(
            [[8, 9], [6, 10], [4, 11], [2, 10], [0, 9], [8, 9], [5, 11], [4, 11], [3, 11], [0, 9], [8, 9], [5, 11],
             [4, 11], [3, 11], [0, 9], [8, 9], [7, 10], [6, 10], [5, 11], [2, 10], [8, 9], [6, 10], [5, 11], [4, 11],
             [1, 10], [7, 10], [5, 11], [4, 11], [3, 11], [1, 10], [7, 10], [4, 11], [3, 11], [2, 10], [0, 9], [6, 10],
             [3, 11], [2, 10], [1, 10], [0, 9]])

        self.assertTrue(torch.equal(input_rules, true_rulebase))

    def test_zero(self):
        x = torch.Tensor([[0, 0, 0, 0, 0],
                          ])
        out = self.anfis(x)

        weights = torch.tensor(
            [[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )

        self.assertTrue(torch.equal(weights, self.anfis.weights))

        normalized_weights = weights
        self.assertTrue(torch.equal(normalized_weights, self.anfis.normalized_weights))

        print(out)
        print(self.anfis.normalized_weights)


if __name__ == '__main__':
    unittest.main()
