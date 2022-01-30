import math

import numpy as np

from utils import wraptopi


def fuzzy_error(curr, tar, future, robot):
    x, y = robot.get_pose()
    current_angle = robot.get_angle()

    pose = np.array([x, y])

    A = np.array([[curr[1] - tar[1], tar[0] - curr[0]], [tar[0] - curr[0], tar[1] - curr[1]]])
    b = np.array([[tar[0] * curr[1] - curr[0] * tar[1]], [x * (tar[0] - curr[0]) + y * (tar[1] - curr[1])]])
    proj = np.matmul(np.linalg.inv(A), b)
    d = (x - curr[0]) * (tar[1] - curr[1]) - (y - curr[1]) * (tar[0] - curr[0])

    side = np.sign(d)

    distance_line = np.linalg.norm(pose - proj.T, 2) * side  ##########################check this
    distance_target = np.linalg.norm(tar - pose)

    k = 0.9

    far_target = np.array([k * proj[0] + (1 - k) * tar[0], k * proj[1] + (1 - k) * tar[1]])
    th1 = math.atan2(far_target[1] - y, far_target[0] - x)
    th2 = math.atan2(tar[1] - curr[1], tar[0] - curr[0])
    th3 = math.atan2(future[1] - tar[1], future[0] - tar[0])

    theta_far = th1 - current_angle
    theta_near = th2 - current_angle
    theta_lookahead = th3 - current_angle

    theta_far = wraptopi(theta_far)
    theta_near = wraptopi(theta_near)
    theta_lookahead = wraptopi(theta_lookahead)

    return [distance_target, distance_line, theta_lookahead, theta_far, theta_near]


def reward(errors, linear_vel, angular_vel, params):
    scale = params['reward_scale']

    DE_penalty_gain = params['DE_penalty_gain']
    DE_penalty_shape = params['DE_penalty_shape']
    HE_penalty_gain = params['HE_penalty_gain']
    HE_penalty_shape = params['HE_penalty_shape']
    HE_iwrt_DE = params['HE_iwrt_DE']
    vel_reward_gain = params['vel_reward_gain']
    vel_iwrt_DE = params['vel_iwrt_DE']
    steering_penalty_gain = params['steering_penalty_gain']
    steering_iwrt_DE = params['steering_iwrt_DE']

    if errors.shape[0] == 5:
        _, dis, _, theta_far, theta_near = errors
    else:
        dis, theta_far, theta_near = errors

    dis_temp = np.abs(dis) / 1.0
    dis = (math.pow(dis_temp, DE_penalty_shape) + dis_temp) * -DE_penalty_gain

    theta_near_temp = np.abs(theta_near) / np.pi
    theta_near_temp = 1 / (1 + np.exp(-25 * theta_near_temp)) - .5
    theta_near_temp *= 2

    theta_near = math.pow(theta_near_temp, HE_penalty_shape) * HE_penalty_gain * (
            1 + 1 / (np.exp(dis_temp * HE_iwrt_DE))) * -15
    theta_near /= 100

    theta_far_temp = np.abs(theta_far) / np.pi
    theta_far_temp = 1 / (1 + np.exp(-4.5 * theta_far_temp)) - .5
    theta_far_temp *= 2
    theta_far = math.pow(theta_far_temp, HE_penalty_shape) * HE_penalty_gain * (
            1 + 1 / (np.exp(dis_temp * HE_iwrt_DE))) * -1.5
    # theta_far /= 6
    theta_far /= 12

    linear_vel = linear_vel * vel_reward_gain / (np.exp(dis_temp * vel_iwrt_DE)) + linear_vel * vel_reward_gain / 2

    angular_vel = np.abs(angular_vel) * steering_penalty_gain / (np.exp(dis_temp * steering_iwrt_DE)) * 1

    rewards = (dis + theta_near + theta_far + linear_vel + angular_vel) / scale
    return rewards, [dis / scale, theta_near / scale, theta_far / scale, linear_vel / scale, angular_vel / scale]
