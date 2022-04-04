import random

import numpy as np
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from gym import spaces
from gym.utils import seeding
from robot_localization.srv import SetPose
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion

from gazebo_utils.path import Path, extend_path
from gazebo_utils.test_course import test_course3
from new_ddpg.gym_gazebo import GazeboEnv
from rl.utils import fuzzy_error


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
    RUNNING = False

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
        # self.odometry = rospy.Subscriber(f'/{self.namespace}/odometry/local_filtered', Odometry, self.odometry_callback,queue_size=1)
        self.odometry = rospy.Subscriber(f'/gazebo/model_states', ModelStates, self.odometry_callback, queue_size=1)

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        # distance_target, distance_line, theta_lookahead, theta_far, theta_near

        self.observation_space = spaces.Box(low=np.array([0., -np.inf, -np.pi, -np.pi, -np.pi]),
                                            high=np.array([np.inf, np.inf, np.pi, np.pi, np.pi]), dtype=np.float32)

        self.action_space = spaces.Box(low=np.array([-JackalState.MAX_ANG_SPEED, 0]),
                                       high=np.array([JackalState.MAX_ANG_SPEED, JackalState.MAX_SPEED]),
                                       dtype=np.float32)
        self.reward_range = (-np.inf, np.inf)

        self.stop = False
        self.read_first_data = 0

        self.max_time = self.path.get_estimated_time(JackalState.MAX_SPEED / 2) * 1.5

        self.start_time = 0
        self.step_iterator = 0

        self.temp_robot = JackalState()

        self._seed()
        self.reset()

    def odometry_callback(self, msg: ModelStates):
        pose = msg.pose[1]

        self.read_first_data += 1
        self.robot.x = pose.position.x
        self.robot.y = pose.position.y

        orientation_q = pose.orientation
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
        rate = rospy.Rate(50)
        if self.step_iterator == 0:
            self.start_time = rospy.get_time()

        # print(action)

        if not GazeboJackalEnv.RUNNING:
            rospy.wait_for_service('/gazebo/unpause_physics')
            try:
                self.unpause()
            except (rospy.ServiceException) as e:
                print("/gazebo/unpause_physics service call failed")
        GazeboJackalEnv.RUNNING = True

        ang, lin = action

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

        # print(self.max_time, dt, state[1])

        if dt > self.max_time or abs(state[1]) > 2:
            done = True
            print("DONE!")

        self.config['steps'] = self.step_iterator

        reward, components = self.reward_fnc(state, lin, ang, self.config)

        self.step_iterator += 1

        rate.sleep()

        return state, reward, done, {"components": components}

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("/gazebo/reset_world service call failed")

        # Resets the state of the environment and returns an initial observation.
        call_service(f'/{self.namespace}/set_pose', SetPose, [])

        vel_cmd = Twist()
        vel_cmd.linear.x = 0
        vel_cmd.angular.z = 0
        self.vel_pub.publish(vel_cmd)

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

        while self.read_first_data < 20 and not rospy.is_shutdown():
            rate.sleep()

        print("Got first data point")
        print(self.robot.get_pose(), self.robot.get_angle())

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
        GazeboJackalEnv.RUNNING = False

        state, _ = self.take_observation()
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
