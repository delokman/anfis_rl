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


# computes the reward for inputs of the given structure
def sub_reward(state, scalar, penalty_gain, add, abs_dis, iwrt_DE):
    """ theta = math.pow(input, HE_penalty_shape)
          ang = np.abs(angular_vel)
      theta_near  = scalar *      theta * HE_penalty_gain       * (1 + 1 / (np.exp(dis_temp * HE_iwrt_DE)))
      linear_vel  = scalar * linear_vel * vel_reward_gain       * (1 + 1 / (np.exp(dis_temp * vel_iwrt_DE)))
      angular_vel = scalar *        ang * steering_penalty_gain * (0 + 1 / (np.exp(dis_temp * steering_iwrt_DE)))"""
    return scalar * state * penalty_gain * (add + 1 / (np.exp(abs_dis * iwrt_DE)))


# Scale inputs for the reward function. Here c stands for coefficient.
def scaled_input(state, c=np.pi):
    return np.abs(state) / c


def scaled_sigmoid(state, c1, c2=np.pi, c3=2, c4=-0.5):
    sigmoid = 1 / (1 + np.exp(-c1 * scaled_input(state, c2)))  # sigmoid function for a scaled input
    output = c3 * (sigmoid + c4)  # offset sigmoid to start at 0 and scale to top out at 1 by default
    return output


def reward(errors, linear_vel, angular_vel, params):
    # scale = params['reward_scale']
    # DE_penalty_gain = params['DE_penalty_gain']
    #vel_reward_gain = params['vel_reward_gain']
    # DE_penalty_shape = params['DE_penalty_shape']
    # HE_penalty_gain = params['HE_penalty_gain']
    # HE_penalty_shape = params['HE_penalty_shape']
    # HE_iwrt_DE = params['HE_iwrt_DE']
    # vel_iwrt_DE = params['vel_iwrt_DE']
    # steering_penalty_gain = params['steering_penalty_gain']
    # steering_iwrt_DE = params['steering_iwrt_DE']
    # dis_scale = params['dis_scale']
    # sigmoid_near = params['sigmoid_near']
    # scale_near = params['scale_near']
    # sigmoid_recovery = params['sigmoid_recovery']
    # scale_recovery = params['scale_recovery']
    # exp_lookahead = params['exp_lookahead']
    # scale_lookahead = params['scale_lookahead']
    # max_angular_vel = params['max_angular_vel']

    if errors.shape[0] == 5:
        # theta recovery = theta far
        target, dis, theta_lookahead, theta_recovery, theta_near = errors
    else:
        dis, theta_recovery, theta_near = errors
        target = 0
        theta_lookahead = 0

    # scaled_dis = scaled_input(dis, dis_scale)
    # dis = -DE_penalty_gain * (math.pow(scaled_dis, DE_penalty_shape) + scaled_dis)
    #
    # scaled_theta_near = scaled_sigmoid(theta_near, sigmoid_near)
    # theta_n = math.pow(scaled_theta_near, HE_penalty_shape)
    # theta_near = sub_reward(theta_n, scale_near, HE_penalty_gain, 1, scaled_dis, HE_iwrt_DE)
    #
    # scaled_theta_recovery = scaled_sigmoid(theta_recovery, sigmoid_recovery)
    # theta_r = math.pow(scaled_theta_recovery, HE_penalty_shape)
    # theta_recovery = sub_reward(theta_r, scale_recovery, HE_penalty_gain, 1, scaled_dis, HE_iwrt_DE)
    #
    # scaled_theta_lh = scaled_input(theta_lookahead)
    # max_turn_r = np.abs(linear_vel) / max_angular_vel  # linear vel / max turn velocity
    # theta_lookahead = scale_lookahead * scaled_theta_lh * np.exp(-exp_lookahead * max_turn_r * target)
    #
    # # vel_iwrt_DE = vel_iwrt_DE * 10
    # vel_iwrt_DE = vel_iwrt_DE
    #
    # linear_vel = sub_reward(linear_vel, 1, vel_reward_gain, 0, scaled_dis, vel_iwrt_DE)  # + \
    # linear_vel ** 2 * np.log((target + 1)) * .5 / 1.5  # / np.exp(linear_vel * vel_iwrt_DE)
    # linear_vel * np.log((target + 1) ** 5.6) * .5 / np.exp(linear_vel * vel_iwrt_DE)

    # abs_angular_vel = np.abs(angular_vel)
    # angular_vel = sub_reward(abs_angular_vel, 1, steering_penalty_gain, 0, scaled_dis,
    #                          steering_iwrt_DE) / 1.5
    # angular_vel = 0

    # dis = 1 / (-dis + 1)
    # theta_near = 1 / (-theta_near + 1)
    # theta_recovery = 1 / (-theta_recovery + 1)
    # theta_lookahead = 1 / (-theta_lookahead + 1)

    # rewards = (dis + theta_near + theta_recovery + linear_vel + angular_vel + theta_lookahead) / scale
    #return rewards, [dis / scale, theta_near / scale, theta_recovery / scale, linear_vel / scale, angular_vel / scale,
    #                 theta_lookahead / scale]
    dis = 1 / (abs(dis) + 0.25)

    rewards = -dis
    return rewards, [dis, theta_near, theta_recovery, linear_vel, angular_vel, theta_lookahead]

