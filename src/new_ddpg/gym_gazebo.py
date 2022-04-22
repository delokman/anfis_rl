import os
import random
import signal
import subprocess
# import roslaunch
import sys

import gym
import rospy
# import time
# from rosgraph_msgs.msg import Clock
from gym.utils import seeding


# From https://github.com/erlerobot/gym-gazebo/blob/master/gym_gazebo/envs/gazebo_env.py


class GazeboEnv(gym.Env):
    """Superclass for all Gazebo environments.
    """
    metadata = {'render.modes': ['human']}

    INITIALIZED = False
    PORT = ""
    PORT_GAZEBO = ""

    def __init__(self, launchfile, ros_path=None, bash=True, arguments=tuple()):
        # self.last_clock_msg = Clock()

        self.np_random = None

        if not GazeboEnv.INITIALIZED:
            random_number = random.randint(10000, 15000)
            # self.port = "11311"#str(random_number) #os.environ["ROS_PORT_SIM"]
            # self.port_gazebo = "11345"#str(random_number+1) #os.environ["ROS_PORT_SIM"]
            GazeboEnv.PORT = str(random_number)  # os.environ["ROS_PORT_SIM"]
            GazeboEnv.PORT_GAZEBO = str(random_number + 1)  # os.environ["ROS_PORT_SIM"]

            os.environ["ROS_MASTER_URI"] = "http://localhost:" + GazeboEnv.PORT
            os.environ["GAZEBO_MASTER_URI"] = "http://localhost:" + GazeboEnv.PORT_GAZEBO
            #
            # self.ros_master_uri = os.environ["ROS_MASTER_URI"];

            print("export ROS_MASTER_URI=http://localhost:" + GazeboEnv.PORT)
            print("export GAZEBO_MASTER_URI=http://localhost:" + GazeboEnv.PORT_GAZEBO)

        # self.port = os.environ.get("ROS_PORT_SIM", "11311")
        if not ros_path:
            ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"]))

        # NOTE: It doesn't make sense to launch a roscore because it will be done when spawing Gazebo, which also need
        #   to be the first node in order to initialize the clock.
        # # start roscore with same python version as current script
        # self._roscore = subprocess.Popen([sys.executable, os.path.join(ros_path, b"roscore"), "-p", self.port])
        # time.sleep(1)
        # print ("Roscore launched!")

        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), launchfile)
        if not os.path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        if bash:
            createGazebo = "0" if GazeboEnv.INITIALIZED else "1"

            command = ['/bin/bash', fullpath, GazeboEnv.PORT, createGazebo, *map(str, arguments)]

            print("Running", " ".join(command))

            # command = " ".join(command)

            environ = {'ROS_DISTRO': 'melodic',
                       'ROS_PACKAGE_PATH': '/home/auvsl/catkin_ws/src:/opt/ros/melodic/share',
                       'HOME': '/home/auvsl',
                       'PATH': '/home/auvsl/catkin_ws/devel/bin:/opt/ros/melodic/bin:/home/auvsl/.local/bin:'
                               '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:'
                               '/usr/local/games:/snap/bin',
                       'CMAKE_PREFIX_PATH': '/home/auvsl/catkin_ws/devel:/opt/ros/melodic', 'DISPLAY': ':0',
                       'SHELL': '/bin/bash',
                       'LD_LIBRARY_PATH': '/home/auvsl/catkin_ws/devel/lib:/opt/ros/melodic/lib',
                       'ROS_PYTHON_VERSION': '2', 'USERNAME': 'auvsl',
                       'PYTHONPATH': '/home/auvsl/catkin_ws/devel/lib/python2.7/dist-packages:/opt/ros/melodic/lib'
                                     '/python2.7/dist-packages',
                       'ROS_ROOT': '/home/auvsl/catkin_ws/src/ros/core/rosbuild',
                       '_': '/usr/bin/python',
                       'ROSLISP_PACKAGE_DIRECTORIES': '/home/auvsl/catkin_ws/devel/share/common-lisp',
                       'ROS_ETC_DIR': '/opt/ros/melodic/etc/ros', 'ROS_VERSION': '1',
                       "ROS_MASTER_URI": "http://localhost:" + GazeboEnv.PORT,
                       "GAZEBO_MASTER_URI": "http://localhost:" + GazeboEnv.PORT_GAZEBO}

            self._roslaunch = subprocess.Popen(command, executable='/bin/bash', env=environ)
        else:
            self._roslaunch = subprocess.Popen(
                [sys.executable, os.path.join(ros_path, b"roslaunch"), "-p", GazeboEnv.PORT, fullpath])
        print("Gazebo launched!")

        self.gzclient_pid = 0

        if not GazeboEnv.INITIALIZED:
            # Launch the simulation with the given launchfile name
            rospy.init_node('gym', anonymous=True)
            GazeboEnv.INITIALIZED = True

        ################################################################################################################
        # r = rospy.Rate(1)
        # self.clock_sub = rospy.Subscriber('/clock', Clock, self.callback, queue_size=1000000)
        # while not rospy.is_shutdown():
        # print("initialization: ", rospy.rostime.is_rostime_initialized())
        # print("Wallclock: ", rospy.rostime.is_wallclock())
        # print("Time: ", time.time())
        # print("Rospyclock: ", rospy.rostime.get_rostime().secs)
        # # print("/clock: ", str(self.last_clock_msg))
        # last_ros_time_ = self.last_clock_msg
        # print("Clock:", last_ros_time_)
        # print("Waiting for synch with ROS clock")
        # if wallclock == False:
        #     break
        # r.sleep()
        ################################################################################################################

    # def callback(self, message):
    #     """
    #     Callback method for the subscriber of the clock topic
    #     :param message:
    #     :return:
    #     """
    #     self.last_clock_msg = int(str(message.clock.secs) + str(message.clock.nsecs)) / 1e6
    #     print("Message", message)
    #     self.last_clock_msg = message
    #     print("Message", message)

    def step(self, action):

        # Implement this method in every subclass
        # Perform a step in gazebo. E.g. move the robot
        raise NotImplementedError

    def reset(self):

        # Implemented in subclass
        raise NotImplementedError

    def render(self, mode="human", close=False):

        if close:
            tmp = os.popen("ps -Af").read()
            proccount = tmp.count('gzclient')
            if proccount > 0:
                if self.gzclient_pid != 0:
                    os.kill(self.gzclient_pid, signal.SIGTERM)
                    os.wait()
            return

        tmp = os.popen("ps -Af").read()
        proccount = tmp.count('gzclient')
        if proccount < 1:
            subprocess.Popen("gzclient")
            self.gzclient_pid = int(subprocess.check_output(["pidof", "-s", "gzclient"]))
        else:
            self.gzclient_pid = 0

    def close(self):

        # Kill gzclient, gzserver and roscore
        tmp = os.popen("ps -Af").read()
        gzclient_count = tmp.count('gzclient')
        gzserver_count = tmp.count('gzserver')
        roscore_count = tmp.count('roscore')
        rosmaster_count = tmp.count('rosmaster')

        if gzclient_count > 0:
            os.system("killall -9 gzclient")
        if gzserver_count > 0:
            os.system("killall -9 gzserver")
        if rosmaster_count > 0:
            os.system("killall -9 rosmaster")
        if roscore_count > 0:
            os.system("killall -9 roscore")

        if gzclient_count or gzserver_count or roscore_count or rosmaster_count > 0:
            os.wait()

    def _configure(self):

        # TODO
        # From OpenAI API: Provides runtime configuration to the enviroment
        # Maybe set the Real Time Factor?
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == '__main__':
    a = random.randint(0, 10000)

    GazeboEnv("/home/auvsl/catkin_ws/src/anfis_rl/multi_simulation.sh", ros_path=r"/opt/ros/melodic/bin/roscore",
              arguments=(str(a),))
