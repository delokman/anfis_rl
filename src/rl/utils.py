import math

import numpy as np

from utils import wraptopi


def fuzzy_error(curr, tar, future, robot):
    x, y = robot.get_pose()
    current_angle = robot.get_angle()

    A = np.array([[curr[1] - tar[1], tar[0] - curr[0]], [tar[0] - curr[0], tar[1] - curr[1]]])
    b = np.array([[tar[0] * curr[1] - curr[0] * tar[1]], [x * (tar[0] - curr[0]) + y * (tar[1] - curr[1])]])
    proj = np.matmul(np.linalg.inv(A), b)
    d = (x - curr[0]) * (tar[1] - curr[1]) - (y - curr[1]) * (tar[0] - curr[0])

    side = np.sign(d)

    distance_line = np.linalg.norm(np.array([x, y]) - proj.T, 2) * side  ##########################check this

    far_target = np.array([0.9 * proj[0] + 0.1 * tar[0], 0.9 * proj[1] + 0.1 * tar[1]])
    th1 = math.atan2(far_target[1] - y, far_target[0] - x)
    th2 = math.atan2(tar[1] - curr[1], tar[0] - curr[0])
    th3 = math.atan2(future[1] - tar[1], future[0] - tar[0])
    theta_far = th1 - current_angle
    theta_near = th2 - current_angle
    theta_far = wraptopi(theta_far)
    theta_near = wraptopi(theta_near)

    return [distance_line, theta_far, theta_near]


def reward(errors, linear_vel, angular_vel):
    DE_penalty_gain = 25
    DE_penalty_shape = 1
    HE_penalty_gain = 25
    HE_penalty_shape = 3
    HE_iwrt_DE = 2
    TDD_reward_gain = 5
    TDD_iwrt_DE = 5
    vel_reward_gain = 1
    vel_iwrt_DE = 1
    steering_penalty_gain = 1
    steering_iwrt_DE = 4

    dis, theta_far, theta_near = errors

    dis_temp = np.abs(dis) / 1.0
    dis = (math.pow(dis_temp, DE_penalty_shape) + dis_temp) * -DE_penalty_gain

    theta_near_temp = theta_near / np.pi
    theta_near = math.pow(theta_near_temp, HE_penalty_shape) * HE_penalty_gain / (np.exp(dis_temp * HE_iwrt_DE)) * -15

    theta_far_temp = np.abs(theta_far) / np.pi
    theta_far = math.pow(theta_far_temp, HE_penalty_shape) * HE_penalty_gain / (np.exp(dis_temp * HE_iwrt_DE)) * -1.5

    linear_vel = linear_vel * vel_reward_gain / (np.exp(dis_temp * vel_iwrt_DE))

    angular_vel = np.abs(angular_vel) * steering_penalty_gain / (np.exp(dis_temp * steering_iwrt_DE)) * -1

    rewards = dis + theta_near + theta_far + linear_vel + angular_vel
    return rewards / 60
