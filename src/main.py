#! /usr/bin/env python
import datetime
import os

import matplotlib

from rl.checkpoint_storage import LowestCheckpoint

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import rospy
import torch
from std_srvs.srv import Empty
from robot_localization.srv import SetPose

from torch.utils.tensorboard import SummaryWriter

from anfis.utils import plot_fuzzy_consequent, plot_fuzzy_membership_functions, plot_fuzzy_variables
from jackal import Jackal
from path import Path
from rl.ddpg import DDPGAgent
from rl.predifined_anfis import predefined_anfis_model
from rl.utils import fuzzy_error, reward
from test_course import test_course, test_course2


def call_service(service_name, service_type, data):
    rospy.wait_for_service(service_name)
    try:
        service = rospy.ServiceProxy(service_name, service_type)
        service(*data)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)


def reset_world():
    call_service('/gazebo/reset_world', Empty, [])
    call_service('/set_pose', SetPose, [])
    rospy.sleep(2)


def plot_anfis_data(epoch, agent):
    anfis = agent.actor

    plot_fuzzy_consequent(summary, anfis, epoch)
    plot_fuzzy_membership_functions(summary, anfis, epoch)
    plot_fuzzy_variables(summary, anfis, epoch)


def epoch(i, agent, path, summary, checkpoint):
    print(f"EPOCH {i}")
    reset_world()

    rate = rospy.Rate(1000)

    jackal = Jackal()
    path = Path(path)

    jackal.wait_for_publisher()

    default_linear_velocity = 1.5

    jackal.linear_velocity = default_linear_velocity

    distance_errors = []
    rewards_cummulative = []

    batch_size = 128
    done = False
    max_yaw_rate = 4
    update_step = 0

    while not rospy.is_shutdown():
        current_point, target_point, future_point, stop = path.get_trajectory(jackal)

        if stop:
            print("STOP")
            break

        path_errors = fuzzy_error(current_point, target_point, future_point, jackal)

        dist_e = path_errors[0]

        if np.isfinite(dist_e):
            distance_errors.append(dist_e)

            if dist_e > 4:
                print("Reloading from save,", checkpoint.checkpoint_location)
                checkpoint.reload(agent)
                return
        else:
            print("Error!!!", path_errors)

        path_errors = np.array(path_errors)

        #   for ddpg model
        control_law = agent.get_action(path_errors)
        control_law = control_law.item()

        if not np.isfinite(control_law):
            print("Error for control law resetting")
            print("Reloading from save,", checkpoint.checkpoint_location)
            checkpoint.reload(agent)

        if control_law > max_yaw_rate:
            control_law = max_yaw_rate
        if control_law < -max_yaw_rate:
            control_law = -max_yaw_rate

        jackal.control_law = control_law

        rewards = reward(path_errors, default_linear_velocity, control_law)
        rewards_cummulative.append(rewards)

        if update_step % 100 == 0:

            # do this every 0.05 s
            # rewards = -1000
            state = agent.curr_states
            agent.curr_states = path_errors

            agent.memory.push(state, control_law, rewards, path_errors, done)  # control_law after gain or before gain?
            if len(agent.memory) > batch_size:
                agent.update(batch_size)

        # print(control_law)
        jackal.pub_motion()
        rate.sleep()

    # plot
    test_path = np.array(path.path)
    robot_path = np.array(jackal.robot_path)

    fig, ax = plt.subplots()
    ax.plot(test_path[:-1, 0], test_path[:-1, 1])
    ax.plot(robot_path[:, 0], robot_path[:, 1])

    distance_errors = np.asarray(distance_errors)

    dist_error_mae = np.mean(np.abs(distance_errors))
    dist_error_rsme = np.sqrt(np.mean(np.power(distance_errors, 2)))

    summary.add_figure("Gazebo/Plot", fig, global_step=i)
    summary.add_scalar("Error/Dist Error MAE", dist_error_mae, global_step=i)
    summary.add_scalar("Error/Dist Error RSME", dist_error_rsme, global_step=i)
    plot_anfis_data(i, agent)

    x = np.arange(0, len(distance_errors))

    fig, ax = plt.subplots()
    ax.plot(x, distance_errors)
    summary.add_figure("Gazebo/Distance Errors", fig, global_step=i)

    fig, ax = plt.subplots()
    ax.plot(x, rewards_cummulative)
    summary.add_figure("Gazebo/Rewards", fig, global_step=i)

    checkpoint_loc = os.path.join(summary.get_logdir(), "checkpoints", f"{i}-{dist_error_mae}.chkp")

    torch.save(agent.state_dict(), checkpoint_loc)

    checkpoint.update(dist_error_mae, checkpoint_loc)

    return dist_error_mae


if __name__ == '__main__':
    rospy.init_node('anfis_rl')

    test_path = test_course2()  ####testcoruse MUST start with 0,0 . Check this out
    test_path.append([1000, 1000])

    name = f'Gazebo RL {datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    print(name)

    summary = SummaryWriter(f'/home/auvsl/python3_ws/src/anfis_rl/runs/{name}')
    os.mkdir(os.path.join(summary.get_logdir(), 'checkpoints'))

    agent = DDPGAgent(3, 1, predefined_anfis_model())
    # agent.load_state_dict(torch.load('input'))
    plot_anfis_data(-1, agent)

    loc = os.path.join(summary.get_logdir(), "checkpoints", f"0.chkp")
    agent.save_checkpoint(loc)

    checkpoint_saver = LowestCheckpoint()

    for i in range(1000):
        epoch(i, agent, test_path, summary, checkpoint_saver)
