import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.noise import NormalActionNoise

from anfis.utils import plot_critic_weights
from gazebo_utils.test_course import test_course3
from main import plot_anfis_data
from new_ddpg.jackal_gym import GazeboJackalEnv
from new_ddpg.policy import TD3Policy
from rl.utils import reward4


def create_env():
    course = test_course3()
    reward_func = reward4
    env = GazeboJackalEnv(path=course, reward_fnc=reward_func)
    time.sleep(10)

    return env


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

        self.obs_name = ["Graphs/Target Error", "Graphs/Distance Error", "Graphs/Theta Lookahead", "Graphs/theta_far",
                         "Graphs/Theta Near"]
        self.act_name = ["Logs/Yaw Rate", "Logs/Velocity"]

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        # plot_critic_weights(summary, model.policy.critic, epoch)
        # plot_critic_weights(summary, model.policy.critic_target, epoch)
        # plot_anfis_data(summary, epoch, agent)

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

        for ob_name, ob in zip(self.obs_name, obs[0]):
            self.logger.record(ob_name, ob, exclude=("stdout"))

        for act_name, act in zip(self.act_name, act[0]):
            self.logger.record(act_name, act, exclude=("stdout"))

        self.logger.dump(self.num_timesteps)

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # plot_critic_weights(summary, model.policy.critic, epoch)
        # plot_critic_weights(summary, model.policy.critic_target, epoch)
        # plot_anfis_data(summary, epoch, agent)



if __name__ == '__main__':
    # env = gym.make("Pendulum-v0")

    env = create_env()

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(TD3Policy, env, action_noise=action_noise, verbose=1, tensorboard_log="./logging/")
    model.learn(total_timesteps=100000, log_interval=1, callback=TensorboardCallback())
    model.save("ddpg_pendulum")
    env = model.get_env()
