import unittest

import numpy as np

from rl.utils import fuzzy_error


class Robot:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle

    def get_pose(self):
        return self.x, self.y

    def get_angle(self):
        return np.deg2rad(self.angle)


class TestUtils(unittest.TestCase):
    def test_fuzzy_errors(self):
        r = Robot(0., 1., 0.)

        curr = np.array([0., 0.])
        tar = np.array([1, 0.])
        fut = np.array([2, 0.])

        distance_target, distance_line, theta_lookahead, theta_far, theta_near = fuzzy_error(curr, tar, fut, r)

        self.assertAlmostEqual(distance_target, np.sqrt(2))
        # self.assertAlmostEqual(distance_line, 1.)
        self.assertAlmostEqual(theta_near, 0.)
        self.assertAlmostEqual(theta_lookahead, 0.0)

        r = Robot(0., -1., 0.)

        curr = np.array([0., 0.])
        tar = np.array([1, 0.])
        fut = np.array([2, 0.])

        distance_target, distance_line, theta_lookahead, theta_far, theta_near = fuzzy_error(curr, tar, fut, r)

        self.assertAlmostEqual(distance_target, np.sqrt(2))
        self.assertAlmostEqual(distance_line, -1.)
        self.assertAlmostEqual(theta_near, 0.)
        self.assertAlmostEqual(theta_lookahead, 0.0)

        r = Robot(0., 2., 30.)

        curr = np.array([0., 0.])
        tar = np.array([1, 0.])
        fut = np.array([1, 1.])

        distance_target, distance_line, theta_lookahead, theta_far, theta_near = fuzzy_error(curr, tar, fut, r)

        self.assertAlmostEqual(distance_target, np.sqrt(2 ** 2 + 1))
        self.assertAlmostEqual(distance_line, 2)
        self.assertAlmostEqual(theta_near, -np.deg2rad(30.))
        self.assertAlmostEqual(theta_lookahead, np.deg2rad(60.))

        r = Robot(1, 0., 90.)

        curr = np.array([0., 0.])
        tar = np.array([0, 1.])
        fut = np.array([1, 1.])

        distance_target, distance_line, theta_lookahead, theta_far, theta_near = fuzzy_error(curr, tar, fut, r)

        self.assertAlmostEqual(distance_target, np.sqrt(2))
        self.assertAlmostEqual(distance_line, -1)
        self.assertAlmostEqual(theta_near, -np.deg2rad(0.))
        self.assertAlmostEqual(theta_lookahead, -np.deg2rad(90.))

        r = Robot(0, 1., 45.)

        curr = np.array([0., 0.])
        tar = np.array([1, 1.])
        fut = np.array([2, 2.])

        distance_target, distance_line, theta_lookahead, theta_far, theta_near = fuzzy_error(curr, tar, fut, r)

        self.assertAlmostEqual(distance_target, 1)
        self.assertAlmostEqual(distance_line, np.sqrt(.5 ** 2 + .5 ** 2))
        self.assertAlmostEqual(theta_near, -np.deg2rad(0.))
        self.assertAlmostEqual(theta_lookahead, -np.deg2rad(0.))

        r = Robot(1, 0., 45.)

        curr = np.array([0., 0.])
        tar = np.array([1, 1.])
        fut = np.array([2, 2.])

        distance_target, distance_line, theta_lookahead, theta_far, theta_near = fuzzy_error(curr, tar, fut, r)

        self.assertAlmostEqual(distance_target, 1)
        self.assertAlmostEqual(distance_line, -np.sqrt(.5 ** 2 + .5 ** 2))
        self.assertAlmostEqual(theta_near, -np.deg2rad(0.))
        self.assertAlmostEqual(theta_lookahead, -np.deg2rad(0.))

        r = Robot(1, 0., 30.)

        curr = np.array([0., 0.])
        tar = np.array([1 * np.cos(np.deg2rad(30)), 1. * np.sin(np.deg2rad(30))])
        fut = tar * 2

        distance_target, distance_line, theta_lookahead, theta_far, theta_near = fuzzy_error(curr, tar, fut, r)

        self.assertAlmostEqual(distance_target,
                               np.sqrt((1 - np.cos(np.deg2rad(30))) ** 2 + np.sin(np.deg2rad(30)) ** 2))
        self.assertAlmostEqual(distance_line, -np.sin(np.deg2rad(30)))
        self.assertAlmostEqual(theta_near, -np.deg2rad(0.))
        self.assertAlmostEqual(theta_lookahead, -np.deg2rad(0.))


if __name__ == '__main__':
    unittest.main()
