import math
from typing import Any, List

import numpy as np

from gazebo_utils.utils import wraptopi


def fuzzy_error(curr: np.ndarray, tar: np.ndarray, future: np.ndarray, robot: Any) -> List[float, float, float,
                                                                                           float, float]:
    """
    Calculates the state errors given the current robot position and the current, future and target waypoints
    :param curr: the current waypoint
    :param tar: the target waypoint that is trying to be reached
    :param future: the next waypoint after the target waypoint
    :param robot: a class that returns the pose and the angle and the current agen state
    :returns:
        - distance_target the euclidian distance between the robot's position and the target position
        - distance_line the shortest perpendicular distance between the curr->tar vector and the robot position
        - theta_lookahead the angle difference between the future angle heading (tar->future) and the agent's heading
        - theta_far the angle difference between the recovery projection point and the agent's heading
        - theta_near the angle difference between the current line segment (curr->tar) and the agent's heading
    """
    x, y = robot.get_pose()
    current_angle = robot.get_angle()

    pose = np.array([x, y])

    b = tar - curr
    a = pose - curr

    proj = np.dot(a, b) / np.dot(b, b) * b + curr

    norm = np.array([-b[1], b[0]])
    norm /= np.linalg.norm(norm)

    distance_line = np.dot(norm, a)
    distance_target = np.linalg.norm(tar - pose)

    k = 0.95

    far_target = k * proj + (1 - k) * tar
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
    dis_scale = params['dis_scale']
    sigmoid_near = params['sigmoid_near']
    scale_near = params['scale_near']
    sigmoid_recovery = params['sigmoid_recovery']
    scale_recovery = params['scale_recovery']
    exp_lookahead = params['exp_lookahead']
    scale_lookahead = params['scale_lookahead']
    max_angular_vel = params['max_angular_vel']

    if len(errors) == 5 or errors.shape[0] == 5:
        # theta recovery = theta far
        target, dis, theta_lookahead, theta_recovery, theta_near = errors
    else:
        dis, theta_recovery, theta_near = errors
        target = 0
        theta_lookahead = 0

    scaled_dis = scaled_input(dis, dis_scale)
    dis = -DE_penalty_gain * (math.pow(scaled_dis, DE_penalty_shape) + scaled_dis)

    scaled_theta_near = scaled_sigmoid(theta_near, sigmoid_near)
    theta_n = math.pow(scaled_theta_near, HE_penalty_shape)
    theta_near = sub_reward(theta_n, scale_near, HE_penalty_gain, 1, scaled_dis, HE_iwrt_DE)

    scaled_theta_recovery = scaled_sigmoid(theta_recovery, sigmoid_recovery)
    theta_r = math.pow(scaled_theta_recovery, HE_penalty_shape)
    theta_recovery = sub_reward(theta_r, scale_recovery, HE_penalty_gain, 1, scaled_dis, HE_iwrt_DE)

    scaled_theta_lh = scaled_input(theta_lookahead)
    max_turn_r = np.abs(linear_vel) / max_angular_vel  # linear vel / max turn velocity
    theta_lookahead = scale_lookahead * scaled_theta_lh * np.exp(-exp_lookahead * max_turn_r * target)

    # vel_iwrt_DE = vel_iwrt_DE * 10
    vel_iwrt_DE = vel_iwrt_DE

    linear_vel = sub_reward(linear_vel, 1, vel_reward_gain, 0, scaled_dis, vel_iwrt_DE)  # + \
    # linear_vel ** 2 * np.log((target + 1)) * .5 / 1.5  # / np.exp(linear_vel * vel_iwrt_DE)
    # linear_vel * np.log((target + 1) ** 5.6) * .5 / np.exp(linear_vel * vel_iwrt_DE)

    abs_angular_vel = np.abs(angular_vel)
    angular_vel = sub_reward(abs_angular_vel, 1, steering_penalty_gain, 0, scaled_dis,
                             steering_iwrt_DE) / 1.5
    angular_vel = 0

    # dis = 1 / (-dis + 1)
    # theta_near = 1 / (-theta_near + 1)
    # theta_recovery = 1 / (-theta_recovery + 1)
    # theta_lookahead = 1 / (-theta_lookahead + 1)

    rewards = (dis + theta_near + theta_recovery + linear_vel + angular_vel + theta_lookahead) / scale
    return rewards, [dis / scale, theta_near / scale, theta_recovery / scale, linear_vel / scale, angular_vel / scale,
                     theta_lookahead / scale]


def reward2(errors, linear_vel, angular_vel, params):
    target, dis, theta_lookahead, theta_recovery, theta_near = errors

    distance_error = 1 - abs(dis) ** 0.25

    min_radius = 0.5
    max_vel = 2

    target_discount = 1 - (target / min_radius) ** 0.5

    if target_discount < 0:
        target_discount = 0

    theta_lookahead_error = 1 - (abs(theta_lookahead) / np.pi) ** (0.5)

    angle_forward_total = theta_lookahead_error * target_discount

    linear_vel_error = (linear_vel / max_vel) ** 0.5

    theta_near_error = 1 - (abs(theta_near) / np.pi) ** 0.3

    distance_error *= 1
    angle_forward_total *= 1
    theta_near_error *= 1
    linear_vel_error *= 1

    reward = distance_error + angle_forward_total + linear_vel_error + theta_near_error

    theta_recovery_error = 0
    angular_vel_error = 0

    return reward, [distance_error, theta_near_error, theta_recovery_error, linear_vel_error, angular_vel_error,
                    angle_forward_total]


def reward3(errors, linear_vel, angular_vel, params):
    target, dis, theta_lookahead, theta_recovery, theta_near = errors
    dis = 1 / (abs(dis) + 0.25)

    return dis, [dis, theta_near, theta_recovery, linear_vel, angular_vel,
                 theta_lookahead]


class Reward:
    def __init__(self):
        self.prev_speed = None
        self.prev_turn_speed = None
        self.prev_step = None
        self.prev_direction_diff = None
        self.prev_normalized_distance_from_route = None

    def reset(self):
        self.prev_speed = None
        self.prev_turn_speed = None
        self.prev_step = None
        self.prev_direction_diff = None
        self.prev_normalized_distance_from_route = None

    def __call__(self, errors, linear_vel, angular_vel, params):
        target, dis, theta_lookahead, theta_recovery, theta_near = errors

        steps = params['steps']

        if self.prev_step is None or steps < self.prev_step:
            self.reset()

        # SPEED REWARD
        MIN_SPEED = 1.
        MAX_SPEED = 2.
        optimal_speed = 2.

        sigma_speed = abs(MAX_SPEED - MIN_SPEED) / 6.
        speed_reward = np.exp(-0.5 * abs(linear_vel - optimal_speed) ** 2 / (sigma_speed ** 2))

        has_speed_dropped = False
        if self.prev_speed is not None:
            if self.prev_speed > linear_vel:
                has_speed_dropped = True

        is_turn_upcoming = target < 0.5
        speed_maintain_bonus = 1
        if has_speed_dropped and not is_turn_upcoming:
            speed_maintain_bonus = min(linear_vel / self.prev_speed, 1.)

        # DISTANCE REWARD
        MAX_DISTANCE = 1
        normalized_distance = dis / MAX_DISTANCE

        sigma = abs(MAX_DISTANCE / 4)
        distance_reward = np.exp(-0.5 * abs(normalized_distance) ** 2 / sigma ** 2)

        distance_reduction_bonus = 1
        if self.prev_normalized_distance_from_route is not None and self.prev_normalized_distance_from_route > normalized_distance:
            distance_reduction_bonus = min(abs(self.prev_normalized_distance_from_route / normalized_distance), 2)

        # HEADING
        heading_reward = np.cos(abs(theta_near)) ** 10

        if abs(theta_near) <= np.deg2rad(20):
            heading_reward = np.cos(abs(theta_near)) ** 4

        has_steering_angle_changed = False
        if abs(angular_vel) > .5:
            has_steering_angle_changed = True

        is_heading_in_right_direction = False
        if abs(theta_near) < np.deg2rad(15):
            is_heading_in_right_direction = True

        steering_angle_maintain_bonus = 1.

        if is_heading_in_right_direction and not has_steering_angle_changed:
            if abs(theta_near) < np.deg2rad(10):
                steering_angle_maintain_bonus *= 1.5
            if abs(theta_near) < np.deg2rad(5):
                steering_angle_maintain_bonus *= 1.5
            if self.prev_direction_diff is not None and abs(self.prev_direction_diff) > abs(theta_near):
                steering_angle_maintain_bonus *= 1.25

        heading_decrease_bonus = 0
        if self.prev_direction_diff is not None:
            if is_heading_in_right_direction:
                if abs(self.prev_direction_diff / dis) > 1:
                    heading_decrease_bonus = min(10, abs(self.prev_direction_diff / dis))

        HC = (10 * heading_reward * steering_angle_maintain_bonus)
        DC = (10 * distance_reward * distance_reduction_bonus)
        SC = (5 * speed_reward * speed_maintain_bonus)

        IC = (HC + DC + SC) ** 2 + (HC * DC * SC)

        error_state = False
        if abs(dis) > MAX_DISTANCE:
            error_state = True
        if abs(theta_near) > np.deg2rad(30):
            error_state = True

        if error_state:
            IC = 1e-3

        # LC = (curve_bonus + intermediate_progress_bonus + heading_decrease_bonus)
        LC = (heading_decrease_bonus)

        self.prev_speed = linear_vel
        self.prev_direction_diff = theta_near
        self.prev_step = steps
        self.prev_normalized_distance_from_route = normalized_distance
        self.prev_turn_speed = angular_vel

        total_reward = max(IC + LC, 1e-3)

        return total_reward, dict(Reward=total_reward, IC=IC, LC=LC, HC=HC, DC=DC, SC=SC,
                                  heading_reward=heading_reward,
                                  steering_angle_maintain_bonus=steering_angle_maintain_bonus,
                                  distance_reward=distance_reward,
                                  distance_reduction_bonus=distance_reduction_bonus, speed_reward=speed_reward,
                                  speed_maintain_bonus=speed_maintain_bonus)
