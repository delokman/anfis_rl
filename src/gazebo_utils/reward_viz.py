import matplotlib.pyplot as plt

plt.rcParams['mpl_toolkits.legacy_colorbar'] = False
import numpy as np

from rl.utils import reward, reward3, reward4
from gazebo_utils.utils import reward_function_grid_visualization


def reward_new(errors, linear_vel, angular_vel, params):
    target, dis, theta_lookahead, theta_recovery, theta_near = errors

    distance_error = 1 - abs(dis) ** 0.25

    min_radius = 0.5
    max_vel = 2

    target_discount = 1 - (target / min_radius) ** 0.5
    theta_lookahead_error = 1 - (abs(theta_lookahead) / np.pi) ** 0.5

    if target_discount < 0:
        target_discount = 0

    angle_forward_total = theta_lookahead_error * target_discount

    linear_vel_error = (linear_vel / max_vel) ** 0.5

    theta_near_error = 1 - (abs(theta_near) / np.pi) ** 0.3

    return distance_error + angle_forward_total + linear_vel_error + theta_near_error


if __name__ == '__main__':
    reward_scales = {
        'reward_scale': 15.,
        'DE_penalty_gain': 25 / 1.5,
        'DE_penalty_shape': 1,
        'HE_penalty_gain': 25 * 2,
        'HE_penalty_shape': 3,
        'HE_iwrt_DE': 2,
        'vel_reward_gain': 2,
        'vel_iwrt_DE': 1,
        'steering_penalty_gain': 4,
        'steering_iwrt_DE': 4,
        'dis_scale': 1,
        'sigmoid_near': 25,
        'scale_near': -15 / 100,
        'sigmoid_recovery': 4.5,
        'scale_recovery': -1.5 / 12,
        'exp_lookahead': 1,
        'scale_lookahead': -100 / 2 / 1.5,
        'max_angular_vel': 4,
    }

    n = 50

    variable_ranges = [
        np.linspace(0., 1., num=n),
        np.linspace(-1., 1., num=n),
        np.linspace(-np.pi, np.pi, num=n),
        np.linspace(-np.pi, np.pi, num=n),
        np.linspace(-np.pi, np.pi, num=n),
        np.linspace(0., 2., num=n),
        np.linspace(-4., 4., num=n),
    ]

    variable_names = ["target", "dis", "theta_lookahead", "theta_recovery", "theta_near", "linear_vel", "angular_vel"]


    def reward_all(target, dis, theta_lookahead, theta_recovery, theta_near, linear_vel, angular_vel):
        errors = np.array([target, dis, theta_lookahead, theta_recovery, theta_near])
        return reward(errors, linear_vel, angular_vel, reward_scales)[0]


    def reward_all2(target, dis, theta_lookahead, theta_recovery, theta_near, linear_vel, angular_vel):
        errors = np.array([target, dis, theta_lookahead, theta_recovery, theta_near])
        return reward4(errors, linear_vel, angular_vel, reward_scales)[0]


    def reward_all3(target, dis, theta_lookahead, theta_recovery, theta_near, linear_vel, angular_vel):
        errors = np.array([target, dis, theta_lookahead, theta_recovery, theta_near])
        return reward3(errors, linear_vel, angular_vel, reward_scales)[0]


    fig, grad_fig = reward_function_grid_visualization(variable_ranges, variable_names, reward_all2)

    plt.show()
