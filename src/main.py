import matplotlib.pyplot as plt
import numpy as np
import rospy

from jackal import Jackal
from path import Path
from rl.ddpg import DDPGAgent
from rl.predifined_anfis import predefined_anfis_model
from rl.utils import fuzzy_error, reward
from test_course import test_course

if __name__ == '__main__':
    rospy.init_node('ANFIS_RL')

    test_path = test_course()  ####testcoruse MUST start with 0,0 . Check this out
    test_path.append([1000, 1000])

    rate = rospy.Rate(100)

    jackal = Jackal()
    path = Path(test_path)

    jackal.wait_for_publisher()

    default_linear_velocity = 1.5

    jackal.linear_velocity = default_linear_velocity

    agent = DDPGAgent(3, 1, predefined_anfis_model())

    distance_errors = []

    batch_size = 64
    done = False

    while not rospy.is_shutdown():
        current_point, target_point, future_point, stop = path.get_trajectory(jackal)

        if stop:
            print("STOP")
            break

        path_errors = fuzzy_error(current_point, target_point, future_point, jackal)
        path_errors = np.array(path_errors)

        #   for ddpg model
        control_law = agent.get_action(path_errors)
        control_law = control_law.item() * 8.

        if control_law > 4.:
            control_law = 4.
        if control_law < -4.:
            control_law = -4.

        jackal.control_law = control_law

        rewards = reward(path_errors, default_linear_velocity, control_law)

        # do this every 0.05 s
        # rewards = -1000
        state = agent.curr_states
        agent.curr_states = path_errors
        agent.memory.push(state, control_law, rewards, path_errors, done)  # control_law after gain or before gain?
        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        print(control_law)
        jackal.pub_motion()
        rate.sleep()

    # plot
    test_path = np.array(test_path)
    robot_path = np.array(jackal.robot_path)
    plt.plot(test_path[:-1, 0], test_path[:-1, 1])
    plt.plot(robot_path[:, 0], robot_path[:, 1])
    plt.show()

    # distance error mean
    plt.show()
    # print(np.mean(dis_error))
