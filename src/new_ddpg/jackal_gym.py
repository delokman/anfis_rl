import random
import time

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from gym import spaces
from gym.utils import seeding
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion

from gazebo_utils.path import Path, extend_path
from gazebo_utils.test_course import test_course3
from new_ddpg.gym_gazebo import GazeboEnv
from rl.utils import fuzzy_error


class JackalState:
    MAX_SPEED = 2
    MAX_ANG_SPEED = 4

    def __init__(self, x=0, y=0, angle=0, linear_speed=0, angular_speed=0):
        self.y = y
        self.x = x
        self.angle = angle
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed

    def copy(self, robot):
        self.x = robot.x
        self.y = robot.y
        self.angle = robot.angle
        self.linear_speed = robot.linear_speed
        self.angular_speed = robot.angular_speed

    def get_pose(self):
        return self.x, self.y

    def get_angle(self):
        return self.angle

    def reset(self):
        self.x = 0
        self.y = 0
        self.angle = 0
        self.linear_speed = 0
        self.angular_speed = 0


class GazeboJackalEnv(GazeboEnv):

    def __init__(self, path: list, reward_fnc, config=None, init_pose=(0, 0)):
        # Launch the simulation with the given launchfile name

        self.namespace_id = random.randint(0, 10000)
        self.namespace = f"jackal{self.namespace_id}"

        GazeboEnv.__init__(self, "/home/auvsl/catkin_ws/src/anfis_rl/multi_simulation.sh",
                           ros_path=r"/opt/ros/melodic/bin/roscore",
                           arguments=(self.namespace, *map(str, init_pose)))

        self.reward_fnc = reward_fnc
        self.path = path
        self.config = config
        self.robot = JackalState()

        extend_path(path)
        self.path = Path(path)

        self.vel_pub = rospy.Publisher(f'/{self.namespace}/jackal_velocity_controller/cmd_vel', Twist, queue_size=5)
        self.odometry = rospy.Subscriber(f'/{self.namespace}/odometry/local_filtered', Odometry, self.odometry_callback,
                                         queue_size=1)

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        # distance_target, distance_line, theta_lookahead, theta_far, theta_near
        self.action_space = spaces.Box(low=np.array([0., -np.inf, -np.pi, -np.pi, -np.pi]),
                                       high=np.array([np.inf, np.inf, np.pi, np.pi, np.pi]), dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)

        self.stop = False
        self.read_first_data = 0

        self.max_time = self.path.get_estimated_time(JackalState.MAX_SPEED / 2)

        self.start_time = 0
        self.step_iterator = 0

        self.temp_robot = JackalState()

        self._seed()
        self.reset()

    def odometry_callback(self, msg: Odometry):
        self.read_first_data += 1
        self.robot.x = msg.pose.pose.position.x
        self.robot.y = msg.pose.pose.position.y

        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        self.robot.angle = yaw
        self.robot.linear_speed = msg.twist.twist.linear.x
        self.robot.angular_speed = msg.twist.twist.angular.z

    def take_observation(self):
        self.temp_robot.copy(self.robot)

        curr, targ, fut, done = self.path.get_trajectory(self.temp_robot)
        errors = fuzzy_error(curr, targ, fut, self.temp_robot)

        # print(self.read_first_data, errors)

        return errors, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if self.step_iterator == 0:
            self.start_time = rospy.get_time()

            rospy.wait_for_service('/gazebo/unpause_physics')
            try:
                self.unpause()
            except (rospy.ServiceException) as e:
                print("/gazebo/unpause_physics service call failed")

        lin, ang = action

        vel_cmd = Twist()
        vel_cmd.linear.x = lin
        vel_cmd.angular.z = ang
        self.vel_pub.publish(vel_cmd)

        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     # resp_pause = pause.call()
        #     self.pause()
        # except (rospy.ServiceException) as e:
        #     print("/gazebo/pause_physics service call failed")

        state, done = self.take_observation()

        dt = rospy.get_time() - self.start_time

        if dt > self.max_time:
            done = True
            print("DONE!")

        reward = self.reward_fnc(state, (lin, ang), self.config)

        self.step_iterator += 1
        return state, reward, done, {}

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        print("RESET Complete")
        self.read_first_data = 0

        rate = rospy.Rate(60)

        while not self.read_first_data and not rospy.is_shutdown():
            rate.sleep()

        print("Got first data point")

        self.path.reset()
        self.robot.reset()
        self.temp_robot.reset()

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        self.stop = False

        self.path.set_initial_state(self.robot)
        self.step_iterator = 0
        self.start_time = 0

        state = self.take_observation()
        return state


if __name__ == '__main__':

    test_path = test_course3()

    env = GazeboJackalEnv(test_path, lambda x, y, z: 1)
    env2 = GazeboJackalEnv(test_path, lambda x, y, z: 1)

    time.sleep(10)

    # r = rospy.Rate(1)
    # count = 0
    # while count < 10 and not rospy.is_shutdown():
    #     print("Count:", count)
    #     count += 1
    #     r.sleep()

    t = Twist()
    t.linear.x = 1
    t.angular.z = .25

    r = rospy.Rate(50)
    while not rospy.is_shutdown():
        env.step((1, .2))
        r.sleep()
