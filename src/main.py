#! /usr/bin/env python
# coding=utf-8
import copy
import datetime
import json
import os
import random
import time
import traceback
from typing import Tuple

import matplotlib
import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from new_test_courses import z_course, straight_line, curved_z
from pauser import BluetoothEStop
from rl.checkpoint_storage import LowestCheckpoint
from rl.noise import OUNoise
from test_course import test_course3
from utils import add_hparams, markdown_rule_table

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import rospy
from std_srvs.srv import Empty
from robot_localization.srv import SetPose

from torch.utils.tensorboard import SummaryWriter

from anfis.utils import plot_fuzzy_consequent, plot_fuzzy_membership_functions, plot_fuzzy_variables, \
    plot_critic_weights
from jackal import Jackal
from path import Path
from rl.ddpg import DDPGAgent
from rl.predifined_anfis import optimized_many_error_predefined_anfis_model_with_velocity
from rl.utils import fuzzy_error, reward

import rospkg

np.random.seed(42)
random.seed(42)
torch.random.manual_seed(42)


def call_service(service_name: str, service_type, data):
    """
    Calls a specific ROS Service

    Args:
        service_name (str): the service name to call
        service_type: the input datatype
        data: the data to call the service with
    """
    rospy.wait_for_service(service_name)
    try:
        service = rospy.ServiceProxy(service_name, service_type)
        service(*data)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def reset_world(is_simulation: bool = False):
    """
    Resets the Gazebo world by calling the /gazebo/reset_world and the /set_pose of the robot and then waits for 2 s

    Args:
        is_simulation (bool): flag to see if this is a simulation environment or not, if it is a simulation environment call the services otherwise just wait 2 second
    """
    if is_simulation:
        call_service('/gazebo/reset_world', Empty, [])
        call_service('/set_pose', SetPose, [])
    rospy.sleep(2)


def plot_anfis_data(summary: SummaryWriter, epoch: int, agent: DDPGAgent):
    """
    Plots the actor's fuzzy consequence layer (the triangle output membership functions), the membership functions (all the input membership trapezoids) as images and then all the associated parameters as scalars

    Args:
        summary:  the summary writer to write the output to
        epoch: the current epoch number, will be the x axis
        agent: the DDPG model
    """
    anfis = agent.actor

    plot_fuzzy_consequent(summary, anfis, epoch)
    plot_fuzzy_membership_functions(summary, anfis, epoch)
    plot_fuzzy_variables(summary, anfis, epoch)


def agent_update(agent: DDPGAgent, batch_size: int, summary: SummaryWriter = None):
    """
    Performs the model update step, in this case performs the DDPG model gradient descent steps

    Args:
        agent: the DDPG or any model to update
        batch_size: the batch size to use to sample from memory
        summary: the summary writer to write some debugging information to, such as actor loss, critic loss and TD
    """
    if len(agent.memory) > batch_size:
        agent.update(batch_size, summary)


def add_to_memory(new_state: np.array, rewards: float, control_law: float, agent: DDPGAgent, done: bool):
    """
    Adds the current state to the agent's memory to be later resampled when running the training update

    Args:
        new_state: the cent state of the jackal
        rewards: the associated reward of the current state
        control_law: the current output action of the model
        agent: the DPPG model
        done: if the model has reached the end of the path
    """
    with torch.no_grad():
        ####do this every 0.075 s
        state = agent.curr_states
        new_state = np.array(new_state)
        agent.curr_states = new_state
        agent.memory.push(state, control_law, rewards, new_state, done)  ########control_law aftergain or before gain?


def summary_and_logging(summary: SummaryWriter, agent: DDPGAgent, params: dict, jackal: Jackal, path: Path,
                        distance_errors: list, theta_far_errors: list, theta_near_errors: list,
                        rewards_cummulative: list, checkpoint: LowestCheckpoint, epoch: int, yaw_rates: list,
                        velocities: list, reward_components: list, rule_weights: list = None,
                        train: bool = True) -> float:
    """
    Runs the end of epoch plotting for all the states and information of the DDPG model useful for the troubleshooting

    Args:
        summary: the summary to write the data
        agent: the DDPG model
        params: the params dictionary that contain the hyper-parameters of the training
        jackal: the Jackal object
        path: the current path of the robot trajectory
        distance_errors: the perpendicular distance errors during the run of the epoch
        theta_far_errors: the theta recovery during the run of the epoch
        theta_near_errors: the theta near errors during the run of the epoch
        rewards_cummulative: the total reward during the run
        checkpoint: the checkpoint of the network, a new checkpoint will be saved if the MAE is smaller than the lowest
        epoch: the current epoch number
        yaw_rates: the yaw rates during the run of the epoch
        velocities: the velocity during the run of the epoch
        reward_components: the individual components of the reward function during the run of the epoch
        rule_weights: the weights of the rules during the run of the epoch
        train: flag to know if the model is currently in training mode

    Returns: the mean distance error of the epoch give the distance_errors

    """
    with torch.no_grad():
        plot_critic_weights(summary, agent, epoch)

        if rule_weights is not None and len(rule_weights) > 0:
            fig, ax = plt.subplots()

            averages = torch.mean(torch.mean(torch.stack(rule_weights), dim=1), dim=0)
            rule_ids = [_ for _ in range(averages.shape[0])]

            if agent.use_cuda:
                averages = averages.cpu().detach().numpy()
            else:
                averages = averages.detach().numpy()

            ax.bar(rule_ids, averages)
            ax.set_xticks(rule_ids)
            ax.set_ylabel('Rule weight')
            ax.set_xlabel('Rule')
            fig.tight_layout()

            summary.add_figure('Rules', fig, global_step=epoch)

        # plot
        test_path = np.array(path.path)
        robot_path = np.array(jackal.inverse_transform_poses(path))

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.plot(test_path[:-1, 0], test_path[:-1, 1])
        del test_path
        ax.plot(robot_path[:, 0], robot_path[:, 1])
        del robot_path
        fig.tight_layout()

        distance_errors = np.asarray(distance_errors)

        dist_error_mae = np.mean(np.abs(distance_errors))
        dist_error_rmse = np.sqrt(np.mean(np.power(distance_errors, 2)))
        avg_velocity = np.mean(velocities)
        print("MAE:", dist_error_mae, "RSME:", dist_error_rmse, "AVG Velocity:", avg_velocity)

        summary.add_figure("Path/Plot", fig, global_step=epoch)
        summary.add_scalar("Error/Dist Error MAE", dist_error_mae, global_step=epoch)
        summary.add_scalar("Error/Dist Error RSME", dist_error_rmse, global_step=epoch)
        summary.add_scalar("Error/Average Velocity", avg_velocity, global_step=epoch)

        if train:
            plot_anfis_data(summary, epoch, agent)

        x = np.arange(0, len(distance_errors))

        fig, ax = plt.subplots()
        ax.plot(x, distance_errors)
        fig.tight_layout()
        summary.add_figure("Graphs/Distance Errors", fig, global_step=epoch)

        fig, ax = plt.subplots()
        ax.plot(x, theta_near_errors)
        fig.tight_layout()
        summary.add_figure("Graphs/Theta Near Errors", fig, global_step=epoch)

        fig, ax = plt.subplots()
        ax.plot(x, theta_far_errors)
        fig.tight_layout()
        summary.add_figure("Graphs/Theta Far Errors", fig, global_step=epoch)

        x = np.arange(0, len(velocities))
        fig, ax = plt.subplots()
        ax.plot(x, velocities)
        fig.tight_layout()
        summary.add_figure("Logs/Velocity", fig, global_step=epoch)

        fig, ax = plt.subplots()
        ax.plot(x, yaw_rates)
        fig.tight_layout()
        summary.add_figure("Logs/Yaw Rate", fig, global_step=epoch)

        x = np.arange(0, len(rewards_cummulative))
        fig, ax = plt.subplots()
        ax.plot(x, rewards_cummulative)
        fig.tight_layout()
        summary.add_figure("Reward/Rewards", fig, global_step=epoch)

        total = sum(rewards_cummulative)
        summary.add_scalar('Error/Total Reward', total, global_step=epoch)

        fig, ax = plt.subplots()
        temp = ax.plot(x, reward_components)
        ax.legend(temp, ('dis', 'theta_near', 'theta_far', 'linear_vel', 'angular_vel', 'theta_lookahead'))
        fig.tight_layout()
        summary.add_figure("Reward/Rewards Components", fig, global_step=epoch)

        if train:
            checkpoint_loc = os.path.join(summary.get_logdir(), "checkpoints", f"{epoch}-{dist_error_mae}.chkp")

            agent.save_checkpoint(checkpoint_loc)

            checkpoint.update(dist_error_mae, checkpoint_loc)

            add_hparams(summary, params, {'hparams/Best MAE': checkpoint.error}, step=epoch)
        return dist_error_rmse, dist_error_mae


def shutdown(summary: SummaryWriter, agent: DDPGAgent, params: dict, jackal: Jackal, path: Path, distance_errors: list,
             theta_far_errors: list, theta_near_errors: list,
             rewards_cummulative: list, checkpoint: LowestCheckpoint, epoch: int, yaw_rates: list, velocities: list,
             reward_components: list, rule_weights: list, train: bool):
    """
    Function to be called when rospy gets terminated/when the program is stopped. This tries to save the state of the unfinished epoch before closing

    Args:
        summary: the summary to write the data
        agent: the DDPG model
        params: the params dictionary that contain the hyper-parameters of the training
        jackal: the Jackal object
        path: the current path of the robot trajectory
        distance_errors: the perpendicular distance errors during the run of the epoch
        theta_far_errors: the theta recovery during the run of the epoch
        theta_near_errors: the theta near errors during the run of the epoch
        rewards_cummulative: the total reward during the run
        checkpoint: the checkpoint of the network, a new checkpoint will be saved if the MAE is smaller than the lowest
        epoch: the current epoch number
        yaw_rates: the yaw rates during the run of the epoch
        velocities: the velocity during the run of the epoch
        reward_components: the individual components of the reward function during the run of the epoch
        rule_weights: the weights of the rules during the run of the epoch
        train: flag to know if the model is currently in training mode
    """
    try:
        print("Shutting down by saving data epoch:", epoch)
        summary_and_logging(summary, agent, params, jackal, path, distance_errors, theta_far_errors, theta_near_errors,
                            rewards_cummulative, checkpoint, epoch, yaw_rates, velocities, reward_components,
                            rule_weights, train)
        summary.close()
    except Exception:
        print("Error saving summary data on shutdown")
        traceback.print_exc()


def epoch(i: int, agent: DDPGAgent, path: Path, summary: SummaryWriter, checkpoint: LowestCheckpoint, params: dict,
          pauser: BluetoothEStop, jackal: Jackal, noise: OUNoise = None, show_gradients: bool = False,
          train: bool = True) -> Tuple[float, bool]:
    """
    Runs a full lap of the path, defined as an epoch using the agent as the controller

    Args:
        i: epoch number
        agent: the DDPG model
        path: the path to travel through
        summary: the summary writer
        checkpoint:
        params:
        pauser:
        jackal:
        noise:
        show_gradients:
        train:

    Returns: tuple containing
        - distance_mae (float): the distance mean absolute error during the run using the perpendicular distance
        - error (bool): If an error occurred during the training of the epoch
    """

    rule_weights = []

    print(f"EPOCH {i}")
    reset_world(params['simulation'])
    pauser.wait_for_publisher()

    rate = rospy.Rate(60)
    #rate_update = rospy.Rate(14)
    rate_update = rospy.Rate(60)
    jackal.clear_pose()
    path = Path(path)
    print("Path Length", path.estimated_path_length)

    jackal.wait_for_publisher()

    with torch.no_grad():
        fast = agent.actor.layer[
            'consequent'].mamdani_defs_vel.get_fast().detach()
        med = agent.actor.layer[
            'consequent'].mamdani_defs_vel.get_medium().detach()
        slow = agent.actor.layer[
            'consequent'].mamdani_defs_vel.get_slow().detach()

        jackal.linear_velocity = min((fast + med + slow) / 3., 0.5)
        velocity = jackal.linear_velocity

        timeout_time = path.get_estimated_time(jackal.linear_velocity) * 1.5
        print("Path Timeout period", timeout_time)

    distance_errors = []
    theta_far_errors = []
    theta_near_errors = []
    rewards_cummulative = []

    reward_components = []

    velocities = []
    yaw_rates = []

    done = False
    max_yaw_rate = 4
    max_velocity = 2
    update_step = 0

    start_time = rospy.get_time()

    sleep_rate = rospy.Rate(60)

    path.set_initial_state(jackal)

    rospy.on_shutdown(
        lambda: shutdown(summary, agent, params, jackal, path, distance_errors, theta_far_errors, theta_near_errors,
                         rewards_cummulative, checkpoint, i, yaw_rates, velocities, reward_components, rule_weights,
                         train))

    if noise is not None:
        noise.reset()

    if show_gradients:
        grad_distribution = {'all': []}
        for name, p in agent.actor.named_parameters():
            grad_distribution[name] = []

    error = False

    with tqdm(total=path.path_length) as pbar:
        while not rospy.is_shutdown():
            s = time.time()

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

                print("Reloading from save,", checkpoint.checkpoint_location)
                checkpoint.reload(agent)
                error = True
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

                max_dist_error = 4
                if not params['simulation'] and dist_e > max_dist_error:
                    print("Reloading from save,", checkpoint.checkpoint_location)
                    if train:
                        checkpoint.reload(agent)
                    error = True
                    break
            else:
                print("Error!!!", path_errors)
                print("Reloading from save,", checkpoint.checkpoint_location)
                if train:
                    checkpoint.reload(agent)
                error = True
                break

            path_errors = np.array(path_errors)

            #   for ddpg model
            control_law = agent.get_action(path_errors)

            if agent.actor.velocity:
                control_law, velocity = control_law

            if rule_weights is not None:
                rule_weights.append(agent.actor.weights.detach())

            if noise is not None:
                control_law = noise.get_action(control_law, update_step)

            control_law = control_law.item() * params['control_mul']

            if not np.isfinite(control_law) or not np.isfinite(velocity):
                print("Error for control law resetting", control_law, "velocity,", velocity)
                print("Reloading from save,", checkpoint.checkpoint_location)
                if train:
                    checkpoint.reload(agent)
                error = True
                break

            if control_law > max_yaw_rate:
                control_law = max_yaw_rate
            elif control_law < -max_yaw_rate:
                control_law = -max_yaw_rate

            if velocity > max_velocity:
                velocity = max_velocity
            elif velocity < -max_velocity:
                velocity = -max_velocity

            yaw_rates.append(control_law)
            velocities.append(velocity)

            jackal.control_law = control_law
            jackal.linear_velocity = velocity

            rewards, comps = reward(path_errors, jackal.linear_velocity, control_law, params)
            rewards_cummulative.append(rewards)
            reward_components.append(comps)

            add_to_memory(path_errors, rewards, (control_law, velocity), agent, done)
            if update_step % params['update_rate'] == 0 and train:
                agent_update(agent, params['batch_size'], summary)

                if show_gradients and len(agent.memory) > params['batch_size']:
                    for name, p in agent.actor.named_parameters():
                        grad_distribution['all'].append(p.grad)
                        grad_distribution[name].append(p.grad)


            # print(control_law)
            jackal.pub_motion()

            if update_step % params['update_rate'] == 0 and train:
                rate_update.sleep()
                rate.last_time = rospy.rostime.get_rostime()
            else:
                rate.sleep()
                rate_update.last_time = rospy.rostime.get_rostime()

            e = time.time()

            hz = 1 / (e - s)
            # print(hz)
            update_step += 1

    del rospy.core._client_shutdown_hooks[-1]

    jackal.linear_velocity = 0
    jackal.control_law = 0
    jackal.pub_motion()

    if show_gradients:
        for name, values in grad_distribution.items():
            dist = torch.stack(values).detach()
            summary.add_histogram(f"Gradients/{name}", dist, global_step=i)
            del dist

    dist_error_rmse, dist_error_mae = summary_and_logging(summary, agent, params, jackal, path, distance_errors, theta_far_errors,
                                         theta_near_errors,
                                         rewards_cummulative, checkpoint, i, yaw_rates, velocities, reward_components,
                                         rule_weights, train)
    min_velocity_training_RMSE = 0.09
    if dist_error_rmse < min_velocity_training_RMSE:
        agent.train_velocity = False
        print("train velocity false")
    else:
        agent.train_velocity = True
        print("train velocity true")

    del jackal, path, distance_errors, theta_far_errors, theta_near_errors, rewards_cummulative, yaw_rates, velocities, reward_components, rule_weights

    return dist_error_mae, error


def extend_path(path: list):
    """
    In order to fix angle errors and index out of bounds, the path is extended by a single segment which is in the same angle as the last segment

    Args:
        path: The path to extend
    """
    before_end, end = np.array(path[-2]), np.array(path[-1])

    diff = (end - before_end)
    diff /= np.linalg.norm(diff)

    after_end = diff * 10 + end

    path.append(after_end)


def is_gazebo_simulation():
    """
    Returns if the current environment is on the Jackal or in a Desktop

    Returns: Returns if the current environment is a simulation or not

    """
    topic_names = set(i for i, _ in rospy.get_published_topics())
    return "/gazebo/model_states" in topic_names


if __name__ == '__main__':
    rospy.init_node('anfis_rl')

    validate = False

    trial_num = 4
    for i in range(trial_num):

        # test_path = test_course()  ####testcoruse MUST start with 0,0 . Check this out
        # test_path = test_course2()  ####testcoruse MUST start with 0,0 . Check this out
        test_path = test_course3()  ####testcoruse MUST start with 0,0 . Check this out
        # test_path = hard_course(400)  ####testcoruse MUST start with 0,0 . Check this out
        # test_path = new_test_course_r_1()  ####testcoruse MUST start with 0,0 . Check this out
        extend_path(test_path)

        name = f'Gazebo RL {datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

        print(name)

        is_simulation = is_gazebo_simulation()

        if is_simulation:
            package_location = '/home/auvsl/python3_ws/src/anfis_rl'
        else:
            rospack = rospkg.RosPack()
            package_location = rospack.get_path('anfis_rl')

        print("Package Location:", package_location)

        summary = SummaryWriter(f'{package_location}/runs/{name}')
        os.mkdir(os.path.join(summary.get_logdir(), 'checkpoints'))

        # agent = DDPGAgent(5, 1, optimized_many_error_predefined_anfis_model(), critic_learning_rate=1e-3, hidden_size=32,
        #                   actor_learning_rate=1e-4)
        # agent = DDPGAgent(5, 1, optimized_many_error_predefined_anfis_model(), critic_learning_rate=1e-3, hidden_size=32,
        #                   actor_learning_rate=1e-4, priority=False)
        agent = DDPGAgent(5, 2, optimized_many_error_predefined_anfis_model_with_velocity(), critic_learning_rate=1e-3,
                          hidden_size=32,
                          actor_learning_rate=1e-4, priority=False)

        # agent.critic.load_state_dict(torch.load(f'{package_location}/critic.weights'))

        # agent.load_state_dict(torch.load('input'))
        plot_anfis_data(summary, -1, agent)
        plot_critic_weights(summary, agent, -1)

        loc = os.path.join(summary.get_logdir(), "checkpoints", f"0.chkp")
        agent.save_checkpoint(loc)

        checkpoint_saver = LowestCheckpoint()

        stop_epoch = 1000

        scheduler1 = ExponentialLR(agent.critic_optimizer, gamma=1, verbose=True)
        scheduler2 = ExponentialLR(agent.actor_optimizer, gamma=1, verbose=True)

        print("Is a simulation:", is_simulation)

        if is_simulation:
            noise = None
            # noise = OUNoise(np.array([
            #     [-4, 4],
            # ]))
        else:
            noise = None

        params = {
            'linear_vel': 2,
            'batch_size': 32,
            'update_rate': 5,
            'epoch_nums': 50,
            'control_mul': 1. if is_simulation else 1.,
            'simulation': is_simulation,
            'actor_decay': scheduler2.gamma,
            'critic_decay': scheduler1.gamma,
            'velocity_controlled': agent.actor.velocity
        }

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

        params.update(reward_scales)

        params.update(agent.input_params)

        with open(os.path.join(summary.get_logdir(), "params.json"), 'w') as file:
            json.dump(params, file)

        pauser = BluetoothEStop()

        jackal = Jackal()

        summary.add_text('Rules', markdown_rule_table(agent.actor))

        memory_backup = copy.deepcopy(agent.memory)

        summary.add_scalar('model/critic_lr', scheduler1.get_last_lr()[0], -1)
        summary.add_scalar('model/actor_lr', scheduler2.get_last_lr()[0], -1)

        error_threshold = 0.03

        train = True
        agent.train_inputs = True

        if validate and is_simulation:
            validation_courses = {"Z Course": z_course(5, 15, 180, 15),
                                  "Straight Line": straight_line(), "Straight Line Mini": straight_line(n=10),
                                  "Curved line 1m 0.5m": curved_z(1, .5, 7), "Curved line 5m 1m": curved_z(5, 1, 7)}

            for k, v in validation_courses.items():
                extend_path(v)

        for i in range(params['epoch_nums']):
            summary.add_scalar('model/learning', train, i)
            mae_error, error_flag = epoch(i, agent, test_path, summary, checkpoint_saver, params, pauser, jackal, noise,
                                          train=train)

            if error_flag:
                agent.memory = copy.deepcopy(memory_backup)
            else:
                memory_backup = copy.deepcopy(agent.memory)

                # if i < stop_epoch:
                if mae_error < error_threshold:
                    if train:
                        print("DISABLED TRAINING")
                        train = False
                        agent.eval()
                else:
                    if not train:
                        print("RE-ENABLED TRAINING")
                        train = True
                        agent.train()

                    scheduler1.step()
                    scheduler2.step()
                # sys.exit()

            if validate:
                if is_simulation and i % 10 == 0:
                    for k, v in validation_courses.items():
                        if isinstance(v, list):
                            v = (v, SummaryWriter(f'{package_location}/runs/{name}/{k}'))
                            validation_courses[k] = v

                        agent.eval()
                        path, val_summary = v
                        epoch(i, agent, path, val_summary, checkpoint_saver, params, pauser, jackal, noise,
                              train=False)
                        agent.train()

            summary.add_scalar('model/critic_lr', scheduler1.get_last_lr()[0], i)
            summary.add_scalar('model/actor_lr', scheduler2.get_last_lr()[0], i)
            summary.add_scalar("model/error", error_flag, i)

        summary.close()

        print("Lowest checkpoint error:", checkpoint_saver.error, ' Error:', checkpoint_saver.checkpoint_location)
