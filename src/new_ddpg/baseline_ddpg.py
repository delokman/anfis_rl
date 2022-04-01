import time
import traceback
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat, Figure
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.td3.td3 import TD3

from anfis.utils import plot_model_weights, plot_fuzzy_variables
from gazebo_utils.test_course import test_course3
from main import plot_anfis_model_data
from new_ddpg.jackal_gym import GazeboJackalEnv
from new_ddpg.policy import TD3Policy
from rl.utils import reward


def create_env():
    course = test_course3()
    reward_func = lambda errors, linear_vel, angular_vel, params: ( 1/ (abs(errors[1]) + .25) + 1 / (abs(errors[-1]) + .4), None)
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

    env = GazeboJackalEnv(path=course, reward_fnc=reward_func, config=reward_scales)
    time.sleep(10)

    return env


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

        self._log_freq = 1

        self._param_log = 100

        self.obs_name = ["Graphs/Target Error", "Graphs/Distance Error", "Graphs/Theta Lookahead", "Graphs/theta_far",
                         "Graphs/Theta Near"]
        self.act_name = ["Logs/Yaw Rate", "Logs/Velocity"]

        self.epoch_num = 0

        self.tb_formatter: Optional[TensorBoardOutputFormat] = None

        self.velocities = None
        self.distance_errors = None
        self.poses = None

    def _on_training_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        output_formats = self.logger.output_formats
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter: TensorBoardOutputFormat = next(
            formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

        plot_model_weights(self.tb_formatter.writer, self.model.policy.critic, -1)
        # plot_model_weights(self.tb_formatter.writer, model.policy.critic_target, -1)
        plot_anfis_model_data(self.tb_formatter.writer, -1, self.model.policy.actor)
        # plot_anfis_model_data(self.tb_formatter.writer, -1, model.policy.actor_target)

    def _on_rollout_start(self) -> None:
        del self.velocities, self.distance_errors, self.poses
        self.velocities = []
        self.distance_errors = []

        self.poses = []

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        self.logger: Logger

        obs = self.locals['new_obs']
        act = self.locals['action']

        robot = self.locals['env'].envs[0].robot

        self.poses.append(robot.get_pose())

        for ob_name, ob in zip(self.obs_name, obs[0]):
            self.logger.record(ob_name, ob, exclude=("stdout"))

            if "Distance Error" in ob_name:
                self.distance_errors.append(ob)

        for act_name, act in zip(self.act_name, act[0]):
            self.logger.record(act_name, act, exclude=("stdout"))

            if "Velocity" in act_name:
                self.velocities.append(act)

        if self.n_calls % self._param_log == 0:
            print(self.num_timesteps)
            plot_fuzzy_variables(self.tb_formatter.writer, self.model.policy.actor, self.num_timesteps)
            self.tb_formatter.writer.flush()

        if self.n_calls % self._log_freq == 0:
            self.logger.dump(self.num_timesteps)

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        plot_model_weights(self.tb_formatter.writer, self.model.policy.critic, self.epoch_num)
        # plot_model_weights(self.tb_formatter.writer, model.policy.critic_target, self.epoch_num)
        plot_anfis_model_data(self.tb_formatter.writer, self.epoch_num, self.model.policy.actor,
                              variable_tensorboard_logging=False)
        # plot_anfis_model_data(self.tb_formatter.writer, self.epoch_num, model.policy.actor_target)

        distance_errors = np.asarray(self.distance_errors)

        max_distance_error = np.max(distance_errors)
        dist_error_mae = np.mean(np.abs(distance_errors))
        dist_error_rmse = np.sqrt(np.mean(np.power(distance_errors, 2)))
        avg_velocity = np.mean(self.velocities)

        output = {"Logs/Dist Error MAE": dist_error_mae,
                  "Logs/Dist Error RSME": dist_error_rmse,
                  "Logs/Dist Error Max": max_distance_error,
                  "Logs/Average Velocity": avg_velocity,
                  }

        excl = {i: None for i in output.keys()}

        self.tb_formatter.write(output, key_excluded=excl, step=self.epoch_num)

        env: GazeboJackalEnv = self.locals['env'].envs[0]
        # plot
        test_path = np.array(env.path.path)
        robot_path = env.path.inverse_transform_poses(self.poses)

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.plot(test_path[:-1, 0], test_path[:-1, 1])
        del test_path
        ax.plot(robot_path[:, 0], robot_path[:, 1])
        del robot_path
        fig.tight_layout()
        self.logger.record("Path/Plot", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close(fig)

        self.epoch_num += 1


if __name__ == '__main__':
    # env = gym.make("Pendulum-v0")

    env = create_env()

    try:
        # The noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        # TODO upgrade python version to 3.9 in order to update baseline class to be able to use SDE noise
        model = TD3(TD3Policy, env, action_noise=action_noise, verbose=1, tensorboard_log="./logging/")
        model.learn(total_timesteps=100000, log_interval=1, callback=TensorboardCallback())
        model.save("ddpg_pendulum")
        env = model.get_env()
    except:
        traceback.print_exc()
        env.close()
    env.close()
