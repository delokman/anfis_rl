import json
import os
from itertools import product

import rospkg
import rospy
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from anfis.utils import plot_critic_weights
from gazebo_utils.jackal import Jackal
from gazebo_utils.pauser import BluetoothEStop
from gazebo_utils.test_course import test_course3
from gazebo_utils.utils import markdown_rule_table
from main import extend_path, epoch, plot_anfis_data
from rl.checkpoint_storage import LowestCheckpoint
from rl.ddpg import DDPGAgent
from rl.predifined_anfis import many_error_predefined_anfis_model


def sequence(parameters, path):
    name = ','.join([f"{k}={v}" for k, v in parameters.items()])

    summary = SummaryWriter(f'{package_location}/{name}')
    os.mkdir(os.path.join(summary.get_logdir(), 'checkpoints'))

    agent = DDPGAgent(5, 1, many_error_predefined_anfis_model(), critic_learning_rate=parameters['critic_lr'],
                      hidden_size=parameters['hidden_size'], actor_learning_rate=parameters['actor_lr'])
    plot_anfis_data(summary, -1, agent)
    plot_critic_weights(summary, agent, -1)

    loc = os.path.join(summary.get_logdir(), "checkpoints", f"0.chkp")
    agent.save_checkpoint(loc)

    checkpoint_saver = LowestCheckpoint()

    scheduler1 = ExponentialLR(agent.critic_optimizer, gamma=parameters['critic_decay'], verbose=True)
    scheduler2 = ExponentialLR(agent.actor_optimizer, gamma=parameters['actor_decay'], verbose=True)

    params = {
        'linear_vel': 1.5,
        'batch_size': parameters['batch_size'],
        'update_rate': 10,
        'epoch_nums': 25,
        'control_mul': 1.,
        'simulation': True,
        'actor_decay': parameters['actor_decay'],
        'critic_decay': parameters['critic_decay']
    }

    params.update(agent.input_params)

    with open(os.path.join(summary.get_logdir(), "params.json"), 'w') as file:
        json.dump(params, file)

    pauser = BluetoothEStop()

    jackal = Jackal()

    summary.add_text('Rules', markdown_rule_table(agent.actor))

    for epoch_num in range(params['epoch_nums']):
        error = epoch(epoch_num, agent, path, summary, checkpoint_saver, params, pauser, jackal)
        scheduler1.step()
        scheduler2.step()

        if epoch_num >= 10 and error >= .15:
            break

    print("Lowest checkpoint error:", checkpoint_saver.error, ' Error:', checkpoint_saver.checkpoint_location)


if __name__ == '__main__':
    parameter_config = {
        'critic_lr': [1e-3, 1e-4, 1e-5],
        'actor_lr': [1e-3, 1e-4, 1e-5, 1e-6],
        'hidden_size': [8, 16, 32, 64, 128],
        'actor_decay': [1, 0.95, 0.9],
        'critic_decay': [1, 0.95, 0.9],
        'batch_size': [32, 64, 128, 258]
    }

    path = test_course3()
    extend_path(path)

    rospy.init_node('hp_tunning_anfis_rl')

    rospack = rospkg.RosPack()
    # package_location = rospack.get_path('anfis_rl')
    package_location = '/home/auvsl/python3_ws/src/anfis_rl/hp_tuning'

    print("Package Location:", package_location)

    param_names = list(parameter_config.keys())
    # zip with parameter names in order to get original property
    param_values = (zip(param_names, x) for x in product(*parameter_config.values()))

    for paramset in param_values:
        # use the dict from iterator of tuples constructor
        kwargs = dict(paramset)

        if kwargs['critic_lr'] > kwargs['actor_lr']:
            print(kwargs)
            sequence(kwargs, path)
