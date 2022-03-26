import time

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from gazebo_utils.test_course import test_course3
from new_ddpg.jackal_gym import GazeboJackalEnv
from new_ddpg.policy import TD3Policy
from rl.utils import reward4


def create_env():
    course = test_course3()
    reward_func = reward4
    env = GazeboJackalEnv(path=course, reward_fnc=reward_func)
    time.sleep(10)

    return env


if __name__ == '__main__':
    # env = gym.make("Pendulum-v0")

    env = create_env()

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(TD3Policy, env, action_noise=action_noise, verbose=1, tensorboard_log="./logging/")
    model.learn(total_timesteps=100000, log_interval=1)
    model.save("ddpg_pendulum")
    env = model.get_env()
