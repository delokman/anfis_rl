#! /usr/bin/env python
# coding=utf-8
import copy
import datetime
import json
import os
import random
import traceback

import matplotlib
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from pauser import BluetoothEStop
from rl.checkpoint_storage import LowestCheckpoint
from utils import add_hparams, markdown_rule_table

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import rospy
import torch
from std_srvs.srv import Empty
from robot_localization.srv import SetPose

from torch.utils.tensorboard import SummaryWriter

from anfis.utils import plot_fuzzy_consequent, plot_fuzzy_membership_functions, plot_fuzzy_variables, \
    plot_critic_weights
from jackal import Jackal
from path import Path
from rl.ddpg import DDPGAgent
from rl.predifined_anfis import predefined_anfis_model, many_error_predefined_anfis_model, \
    optimized_many_error_predefined_anfis_model
from rl.utils import fuzzy_error, reward
from test_course import test_course, test_course2, hard_course, test_course3

import rospkg

np.random.seed(42)
random.seed(42)
torch.random.manual_seed(42)


def call_service(service_name, service_type, data):
    rospy.wait_for_service(service_name)
    try:
        service = rospy.ServiceProxy(service_name, service_type)
        service(*data)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def reset_world(is_simulation=False):
    if is_simulation:
        call_service('/gazebo/reset_world', Empty, [])
        call_service('/set_pose', SetPose, [])
    rospy.sleep(2)


def plot_anfis_data(summary, epoch, agent):
    anfis = agent.actor

    plot_fuzzy_consequent(summary, anfis, epoch)
    plot_fuzzy_membership_functions(summary, anfis, epoch)
    plot_fuzzy_variables(summary, anfis, epoch)


def agent_update(agent, batch_size, dis_error, rule_weights=None):
    if len(agent.memory) > batch_size:
        agent.update(batch_size)

        if rule_weights is not None:
            rule_weights.append(agent.actor.weights)


def add_to_memory(new_state, rewards, control_law, agent, done):
    ####do this every 0.075 s
    state = agent.curr_states
    new_state = np.array(new_state)
    agent.curr_states = new_state
    agent.memory.push(state, control_law, rewards, new_state, done)  ########control_law aftergain or before gain?


def summary_and_logging(summary, agent, params, jackal, path, distance_errors, theta_far_errors, theta_near_errors,
                        rewards_cummulative,
                        checkpoint, epoch, rule_weights=None):
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
    ax.plot(robot_path[:, 0], robot_path[:, 1])
    fig.tight_layout()

    distance_errors = np.asarray(distance_errors)

    dist_error_mae = np.mean(np.abs(distance_errors))
    dist_error_rsme = np.sqrt(np.mean(np.power(distance_errors, 2)))
    print("MAE:", dist_error_mae, "RSME:", dist_error_rsme)

    summary.add_figure("Path/Plot", fig, global_step=epoch)
    summary.add_scalar("Error/Dist Error MAE", dist_error_mae, global_step=epoch)
    summary.add_scalar("Error/Dist Error RSME", dist_error_rsme, global_step=epoch)
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

    x = np.arange(0, len(rewards_cummulative))
    fig, ax = plt.subplots()
    ax.plot(x, rewards_cummulative)
    fig.tight_layout()
    summary.add_figure("Graphs/Rewards", fig, global_step=epoch)

    checkpoint_loc = os.path.join(summary.get_logdir(), "checkpoints", f"{epoch}-{dist_error_mae}.chkp")

    agent.save_checkpoint(checkpoint_loc)

    checkpoint.update(dist_error_mae, checkpoint_loc)

    add_hparams(summary, params, {'hparams/Best MAE': checkpoint.error}, step=epoch)
    return dist_error_mae


def shutdown(summary, agent, params, jackal, path, distance_errors, theta_far_errors, theta_near_errors,
             rewards_cummulative, checkpoint, epoch, rule_weights):
    try:
        print("Shutting down by saving data epoch:", epoch)
        summary_and_logging(summary, agent, params, jackal, path, distance_errors, theta_far_errors, theta_near_errors,
                            rewards_cummulative, checkpoint, epoch, rule_weights)
    except Exception:
        print("Error saving summary data on shutdown")
        traceback.print_exc()


def epoch(i, agent, path, summary, checkpoint, params, pauser, jackal, noise=None, show_gradients=False):
    rule_weights = []

    print(f"EPOCH {i}")
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

    done = False
    max_yaw_rate = 4
    update_step = 0

    start_time = rospy.get_time()

    sleep_rate = rospy.Rate(60)

    path.set_initial_state(jackal)

    rospy.on_shutdown(
        lambda: shutdown(summary, agent, params, jackal, path, distance_errors, theta_far_errors, theta_near_errors,
                         rewards_cummulative, checkpoint, i, rule_weights))

    if noise is not None:
        noise.reset()

    if show_gradients:
        grad_distribution = {'all': []}
        for name, p in agent.actor.named_parameters():
            grad_distribution[name] = []

    error = False

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

                if not params['simulation'] and dist_e > 4:
                    print("Reloading from save,", checkpoint.checkpoint_location)
                    checkpoint.reload(agent)
                    error = True
                    break
            else:
                print("Error!!!", path_errors)
                print("Reloading from save,", checkpoint.checkpoint_location)
                checkpoint.reload(agent)
                error = True
                break

            path_errors = np.array(path_errors)

            #   for ddpg model
            control_law = agent.get_action(path_errors)

            if noise is not None:
                control_law = noise.get_action(control_law, update_step)

            control_law = control_law.item() * params['control_mul']

            if not np.isfinite(control_law):
                print("Error for control law resetting")
                print("Reloading from save,", checkpoint.checkpoint_location)
                checkpoint.reload(agent)
                error = True
                break

            if control_law > max_yaw_rate:
                control_law = max_yaw_rate
            if control_law < -max_yaw_rate:
                control_law = -max_yaw_rate

            jackal.control_law = control_law

            rewards = reward(path_errors, jackal.linear_velocity, control_law) / 15.
            rewards_cummulative.append(rewards)

            add_to_memory(path_errors, rewards, control_law, agent, done)
            if update_step % params['update_rate'] == 0:
                agent_update(agent, params['batch_size'], dist_e, rule_weights)

                if show_gradients and len(agent.memory) > params['batch_size']:
                    for name, p in agent.actor.named_parameters():
                        grad_distribution['all'].append(p.grad)
                        grad_distribution[name].append(p.grad)

            update_step += 1
            # print(control_law)
            jackal.pub_motion()
            rate.sleep()

    del rospy.core._client_shutdown_hooks[-1]

    jackal.linear_velocity = 0
    jackal.control_law = 0
    jackal.pub_motion()

    if show_gradients:
        for name, values in grad_distribution.items():
            dist = torch.stack(values)
            summary.add_histogram(f"Gradients/{name}", dist, global_step=i)

    dist_error_mae = summary_and_logging(summary, agent, params, jackal, path, distance_errors, theta_far_errors,
                                         theta_near_errors,
                                         rewards_cummulative, checkpoint, i, rule_weights)

    return dist_error_mae, error


def extend_path(path):
    before_end, end = np.array(path[-2]), np.array(path[-1])

    diff = (end - before_end)
    diff /= np.linalg.norm(diff)

    after_end = diff * 10 + end

    path.append(after_end)


def is_gazebo_simulation():
    topic_names = set(i for i, _ in rospy.get_published_topics())
    return "/gazebo/model_states" in topic_names


if __name__ == '__main__':
    rospy.init_node('anfis_rl')

    # test_path = test_course2()  ####testcoruse MUST start with 0,0 . Check this out
    # test_path = test_course()  ####testcoruse MUST start with 0,0 . Check this out
    # test_path = hard_course(400)  ####testcoruse MUST start with 0,0 . Check this out
    test_path = test_course3()  ####testcoruse MUST start with 0,0 . Check this out
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
    agent = DDPGAgent(5, 1, optimized_many_error_predefined_anfis_model(), critic_learning_rate=1e-2, hidden_size=32,
                      actor_learning_rate=1e-3)
    # agent.critic.load_state_dict(torch.load(f'{package_location}/critic.weights'))

    # agent.load_state_dict(torch.load('input'))
    plot_anfis_data(summary, -1, agent)
    plot_critic_weights(summary, agent, -1)

    loc = os.path.join(summary.get_logdir(), "checkpoints", f"0.chkp")
    agent.save_checkpoint(loc)

    checkpoint_saver = LowestCheckpoint()

    stop_epoch = 14

    scheduler1 = ExponentialLR(agent.critic_optimizer, gamma=.95, verbose=True)
    scheduler2 = ExponentialLR(agent.actor_optimizer, gamma=.95, verbose=True)

    print("Is a simulation:", is_simulation)

    if is_simulation:
        noise = None
        # noise = OUNoise(np.array([
        #     [-4, 4],
        # ]))
    else:
        noise = None

    params = {
        'linear_vel': 1.5,
        'batch_size': 32,
        'update_rate': 1,
        'epoch_nums': 100,
        'control_mul': 1. if is_simulation else 1.,
        'simulation': is_simulation,
        'actor_decay': scheduler2.gamma,
        'critic_decay': scheduler1.gamma
    }

    params.update(agent.input_params)

    with open(os.path.join(summary.get_logdir(), "params.json"), 'w') as file:
        json.dump(params, file)

    pauser = BluetoothEStop()

    jackal = Jackal()

    summary.add_text('Rules', markdown_rule_table(agent.actor))

    memory_backup = copy.deepcopy(agent.memory)

    summary.add_scalar('model/critic_lr', scheduler1.get_last_lr()[0], -1)
    summary.add_scalar('model/actor_lr', scheduler2.get_last_lr()[0], -1)

    error_threshold = 0.075

    train = True

    for i in range(params['epoch_nums']):
        epoch(i, agent, test_path, summary, checkpoint_saver, params, pauser, jackal, noise)
        # if i < stop_epoch:
        scheduler1.step()
        scheduler2.step()
        # sys.exit()

    print("Lowest checkpoint error:", checkpoint_saver.error, ' Error:', checkpoint_saver.checkpoint_location)
