import unittest

import numpy as np

from gazebo_utils.path import Path


class Robot:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle

    def get_pose(self):
        return self.x, self.y

    def get_angle(self):
        return np.deg2rad(self.angle)


class TestPath(unittest.TestCase):
    def test_straight_path(self):
        points = [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [3.1, 0],
            [4, 0],
            [5, 0],  # Ignored point
        ]

        path = Path(points)

        time = path.get_estimated_time(.5)

        self.assertEqual(time, 8)

        robot = Robot(0.5, 0, 0)

        curr, tar, future, done = path.get_trajectory(robot)
        self.assertTrue(np.allclose(curr, np.array([0, 0])))
        self.assertTrue(np.allclose(tar, np.array([1, 0])))
        self.assertTrue(np.allclose(future, np.array([2, 0])))
        self.assertFalse(done)

        robot = Robot(0.5, 1, 0)

        curr, tar, future, done = path.get_trajectory(robot)
        self.assertTrue(np.allclose(curr, np.array([0, 0])))
        self.assertTrue(np.allclose(tar, np.array([1, 0])))
        self.assertTrue(np.allclose(future, np.array([2, 0])))
        self.assertFalse(done)

    def test_overflow(self):
        points = [
            [0, 0],
            [1, 0],
            [5, 0],  # Ignored point
        ]

        path = Path(points)
        robot = Robot(6, 1, 0)

        curr, tar, future, done = path.get_trajectory(robot)
        self.assertTrue(np.allclose(curr, np.array([1, 0])))
        self.assertTrue(np.allclose(tar, np.array([5, 0])))
        self.assertTrue(np.allclose(future, np.array([5, 0])))
        self.assertTrue(done)

    def test_angle(self):
        points = [
            [0, 0],
            [1, 0],
            [1, 1],
            [1, 2],  # Ignored point
        ]

        path = Path(points)
        robot = Robot(2, -3, 0)

        curr, tar, future, done = path.get_trajectory(robot)
        self.assertTrue(np.allclose(curr, np.array([1, 0])))
        self.assertTrue(np.allclose(tar, np.array([1, 1])))
        self.assertTrue(np.allclose(future, np.array([1, 2])))
        self.assertFalse(done)
