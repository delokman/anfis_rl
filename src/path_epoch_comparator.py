#! /usr/bin/env python
# coding=utf-8


import datetime
import glob
import json
import os
import random
import traceback

import matplotlib
from tqdm import tqdm

from gazebo_utils.pauser import BluetoothEStop
from gazebo_utils.utils import markdown_rule_table
from main import is_gazebo_simulation, extend_path, plot_anfis_data, reset_world

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import rospy
import torch

from torch.utils.tensorboard import SummaryWriter

from anfis.utils import plot_critic_weights
from gazebo_utils.jackal import Jackal
from gazebo_utils.path import Path
from rl.ddpg import DDPGAgent
from rl.predifined_anfis import optimized_many_error_predefined_anfis_model
from rl.utils import fuzzy_error, reward
from gazebo_utils.test_course import test_course, test_course2, hard_course, test_course3, test_8_shape

import rospkg

np.random.seed(42)
random.seed(42)
torch.random.manual_seed(42)


def shutdown(summary, agent, jackal, path, distance_errors, theta_far_errors, theta_near_errors,
             rewards_cummulative):
    try:
        print("Shutting down by saving data epoch:")
        summary_plotting(summary, agent, jackal, path, distance_errors, theta_far_errors, theta_near_errors,
                         rewards_cummulative)
    except Exception:
        print("Error saving summary data on shutdown")
        traceback.print_exc()


def summary_plotting(summary, agent, jackal, path, distance_errors, theta_far_errors, theta_near_errors,
                     rewards_cummulative):
    plot_critic_weights(summary, agent, None)

    # plot
    test_path = np.array(path.path)
    robot_path = np.array(jackal.inverse_transform_poses(path))

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.plot(test_path[:-1, 0], test_path[:-1, 1])
    ax.plot(robot_path[:, 0], robot_path[:, 1])
    fig.tight_layout()

    distance_errors = np.asarray(distance_errors)

    dist_error_mae = np.mean(np.abs(distance_errors))
    dist_error_rsme = np.sqrt(np.mean(np.power(distance_errors, 2)))
    print("MAE:", dist_error_mae, "RSME:", dist_error_rsme)

    summary.add_figure("Path/Plot", fig)
    summary.add_scalar("Error/Dist Error MAE", dist_error_mae)
    summary.add_scalar("Error/Dist Error RSME", dist_error_rsme)
    plot_anfis_data(summary, None, agent)

    x = np.arange(0, len(distance_errors))

    fig, ax = plt.subplots()
    ax.plot(x, distance_errors)
    fig.tight_layout()
    summary.add_figure("Graphs/Distance Errors", fig)

    fig, ax = plt.subplots()
    ax.plot(x, theta_near_errors)
    fig.tight_layout()
    summary.add_figure("Graphs/Theta Near Errors", fig)

    fig, ax = plt.subplots()
    ax.plot(x, theta_far_errors)
    fig.tight_layout()
    summary.add_figure("Graphs/Theta Far Errors", fig)

    x = np.arange(0, len(rewards_cummulative))
    fig, ax = plt.subplots()
    ax.plot(x, rewards_cummulative)
    fig.tight_layout()
    summary.add_figure("Graphs/Rewards", fig)


def run_path(agent, path, summary, params, pauser, jackal):
    reset_world(params['simulation'])
    pauser.wait_for_publisher()

    rate = rospy.Rate(60)

    jackal.clear_pose()
    path = Path(path)
    print("Path Length", path.estimated_path_length)

    jackal.wait_for_publisher()

    jackal.linear_velocity = params['linear_vel']

    timeout_time = path.get_estimated_time(jackal.linear_velocity) * 1.5
    print("Path Timeout period", timeout_time)

    distance_errors = []
    theta_far_errors = []
    theta_near_errors = []
    rewards_cummulative = []

    max_yaw_rate = 4

    start_time = rospy.get_time()

    sleep_rate = rospy.Rate(60)

    path.set_initial_state(jackal)

    rospy.on_shutdown(
        lambda: shutdown(summary, agent, jackal, path, distance_errors, theta_far_errors, theta_near_errors,
                         rewards_cummulative))

    with tqdm(total=path.path_length) as pbar:
        while not rospy.is_shutdown():
            while pauser.pause:
                if not rospy.is_shutdown():
                    sleep_rate.sleep()
                else:
                    rospy.signal_shutdown("Requesting shutdown but in e-stop mode")

            current_point, target_point, future_point, stop = path.get_trajectory(jackal, pbar)

            if stop:
                print("STOP")
                break

            difference = rospy.get_time() - start_time

            if difference >= timeout_time:
                print("Exceeded timeout returning to checkpoint")
                break

            path_errors = fuzzy_error(current_point, target_point, future_point, jackal)

            if len(path_errors) == 3:
                dist_e, theta_far, theta_near = path_errors
            elif len(path_errors) == 5:
                _, dist_e, _, theta_far, theta_near = path_errors

            if np.isfinite(dist_e):
                distance_errors.append(dist_e)
                theta_far_errors.append(theta_far)
                theta_near_errors.append(theta_near)

                if not params['simulation'] and dist_e > 4:
                    print("Ran out of course stopping")
                    break
            else:
                print("Error!!!", path_errors)
                break

            path_errors = np.array(path_errors)

            #   for ddpg model
            control_law = agent.get_action(path_errors)

            control_law = control_law.item() * params['control_mul']

            if not np.isfinite(control_law):
                print("Error for control law resetting")

            if control_law > max_yaw_rate:
                control_law = max_yaw_rate
            if control_law < -max_yaw_rate:
                control_law = -max_yaw_rate

            jackal.control_law = control_law

            rewards = reward(path_errors, jackal.linear_velocity, control_law) / 15.
            rewards_cummulative.append(rewards)

            jackal.pub_motion()
            rate.sleep()

    del rospy.core._client_shutdown_hooks[-1]

    jackal.linear_velocity = 0
    jackal.control_law = 0
    jackal.pub_motion()

    summary_plotting(summary, agent, jackal, path, distance_errors, theta_far_errors,
                     theta_near_errors, rewards_cummulative)


def find_first_best_checkpoint(run_directory):
    checkpoint_path = os.path.join(run_directory, 'checkpoints/*.chkp')

    chkps = glob.glob(checkpoint_path)

    chkps = sorted(chkps, key=lambda x: (int(os.path.basename(x).replace(".chkp", '').split('-')[0]), len(x)))

    first = chkps[0]
    del chkps[0]

    def error_epoch(x):
        x = os.path.basename(x).replace(".chkp", '').split('-')

        return float(x[1]), int(x[0])

    chkps = sorted(chkps, key=error_epoch)

    return first, chkps[0]


if __name__ == '__main__':
    rospy.init_node('path_epoch_comparator')

    is_simulation = is_gazebo_simulation()

    if is_simulation:
        package_location = '/home/auvsl/python3_ws/src/anfis_rl'
    else:
        rospack = rospkg.RosPack()
        package_location = rospack.get_path('anfis_rl')

    print("Package Location:", package_location)

    run_name = 'Gazebo RL 2021-10-01-18-39-43'

    run_directory = f'{package_location}/runs/{run_name}'

    first_ckp, best_ckp = find_first_best_checkpoint(run_directory)

    print("Using INITIAL:", first_ckp)
    print("Using BEST:", best_ckp)

    path_defintions = [test_course, test_course2, test_course3, test_8_shape, hard_course]

    pauser = BluetoothEStop()

    jackal = Jackal()

    for path_def in path_defintions:
        path = path_def()
        extend_path(path)
        name = path_def.__name__

        print("Running comparison on:", name)

        for chkp_type, chp in zip(['INITIAL', 'BEST'], [first_ckp, best_ckp]):
            agent = DDPGAgent(5, 1, optimized_many_error_predefined_anfis_model(), critic_learning_rate=1e-3,
                              hidden_size=32,
                              actor_learning_rate=1e-4)
            agent.load_checkpoint(chp)
            agent.eval()

            out_name = f'{name}-{chkp_type}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
            summary = SummaryWriter(f'{package_location}/compare/{out_name}')
            print("Current logdir:", summary.get_logdir())

            params = {
                'linear_vel': 1.5,
                'batch_size': 32,
                'update_rate': 1,
                'epoch_nums': 100,
                'control_mul': 1. if is_simulation else 1.,
                'simulation': is_simulation,
            }

            params.update(agent.input_params)

            with open(os.path.join(summary.get_logdir(), "params.json"), 'w') as file:
                json.dump(params, file)

            plot_anfis_data(summary, -1, agent)
            plot_critic_weights(summary, agent, -1)

            summary.add_text('Rules', markdown_rule_table(agent.actor))

            print("Running path")
            run_path(agent, path, summary, params, pauser, jackal)
