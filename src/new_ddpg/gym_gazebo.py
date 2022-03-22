import os
import random
import signal
import subprocess
# import roslaunch
import sys
import time

import gym
import rospy
from rosgraph_msgs.msg import Clock


class GazeboEnv(gym.Env):
    """Superclass for all Gazebo environments.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, launchfile, ros_path=None, bash=True, arguments=tuple()):
        self.last_clock_msg = Clock()

        random_number = random.randint(10000, 15000)
        # self.port = "11311"#str(random_number) #os.environ["ROS_PORT_SIM"]
        # self.port_gazebo = "11345"#str(random_number+1) #os.environ["ROS_PORT_SIM"]
        self.port = str(random_number)  # os.environ["ROS_PORT_SIM"]
        self.port_gazebo = str(random_number + 1)  # os.environ["ROS_PORT_SIM"]

        os.environ["ROS_MASTER_URI"] = "http://localhost:" + self.port
        os.environ["GAZEBO_MASTER_URI"] = "http://localhost:" + self.port_gazebo
        #
        # self.ros_master_uri = os.environ["ROS_MASTER_URI"];

        print("ROS_MASTER_URI=http://localhost:" + self.port + "\n")
        print("GAZEBO_MASTER_URI=http://localhost:" + self.port_gazebo + "\n")

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
            command = ['/bin/bash', fullpath, self.port, *map(str, arguments)]

            print("Running", " ".join(command))

            # command = " ".join(command)

            environ = {'SNAP_REAL_HOME': '/home/auvsl', 'ROS_DISTRO': 'melodic',
                       'SNAP_LIBRARY_PATH': '/var/lib/snapd/lib/gl:/var/lib/snapd/lib/gl32:/var/lib/snapd/void',
                       'QT4_IM_MODULE': 'xim', 'SNAP_CONTEXT': 'KVHmahN_WPi0HfHdI7xqUdlL8cEgMH2uTesvW0a3N4TcvWrtL-u1',
                       'GJS_DEBUG_OUTPUT': 'stderr', 'LESSOPEN': '| /usr/bin/lesspipe %s',
                       'XDG_CURRENT_DESKTOP': 'ubuntu:GNOME', 'WINDOWPATH': '1',
                       'TERMINAL_EMULATOR': 'JetBrains-JediTerm', 'XDG_SESSION_TYPE': 'x11',
                       'BAMF_DESKTOP_FILE_HINT': '/var/lib/snapd/desktop/applications/pycharm-professional_pycharm-professional.desktop',
                       'QT_IM_MODULE': 'ibus', 'LOGNAME': 'auvsl', 'USER': 'auvsl',
                       'ROS_PACKAGE_PATH': '/home/auvsl/catkin_ws/src:/opt/ros/melodic/share', 'XDG_VTNR': '1',
                       'HOME': '/home/auvsl',
                       'PATH': '/home/auvsl/catkin_ws/devel/bin:/opt/ros/melodic/bin:/home/auvsl/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin',
                       'CMAKE_PREFIX_PATH': '/home/auvsl/catkin_ws/devel:/opt/ros/melodic', 'DISPLAY': ':0',
                       'SNAP_ARCH': 'amd64', 'SSH_AGENT_PID': '1563', 'LANG': 'en_US.UTF-8', 'SNAP_REVISION': '278',
                       'TERM': 'xterm-256color', 'SHELL': '/bin/bash', 'XAUTHORITY': '/run/user/1000/gdm/Xauthority',
                       'SESSION_MANAGER': 'local/auvsl-ubuntu18:@/tmp/.ICE-unix/1467,unix/auvsl-ubuntu18:/tmp/.ICE-unix/1467',
                       'XDG_DATA_DIRS': '/usr/share/ubuntu:/usr/local/share/:/usr/share/:/var/lib/snapd/desktop',
                       'PKG_CONFIG_PATH': '/home/auvsl/catkin_ws/devel/lib/pkgconfig:/opt/ros/melodic/lib/pkgconfig',
                       'MANDATORY_PATH': '/usr/share/gconf/ubuntu.mandatory.path', 'QT_ACCESSIBILITY': '1',
                       'LD_LIBRARY_PATH': '/home/auvsl/catkin_ws/devel/lib:/opt/ros/melodic/lib',
                       'GNOME_DESKTOP_SESSION_ID': 'this-is-deprecated', 'CLUTTER_IM_MODULE': 'xim',
                       'TEXTDOMAIN': 'im-config', 'EDITOR': 'nano -w',
                       'SNAP_USER_DATA': '/home/auvsl/snap/pycharm-professional/278', 'XMODIFIERS': '@im=ibus',
                       'GIO_LAUNCHED_DESKTOP_FILE_PID': '2039', 'XDG_RUNTIME_DIR': '/run/user/1000',
                       'GPG_AGENT_INFO': '/run/user/1000/gnupg/S.gpg-agent:0:1', 'ROS_PYTHON_VERSION': '2',
                       'SNAP_VERSION': '2021.3.3', 'USERNAME': 'auvsl', 'XDG_SESSION_DESKTOP': 'ubuntu',
                       'TERM_SESSION_ID': 'cea573ec-626d-4900-8cf3-5b9095a70c88',
                       'GIO_LAUNCHED_DESKTOP_FILE': '/var/lib/snapd/desktop/applications/pycharm-professional_pycharm-professional.desktop',
                       'SNAP_DATA': '/var/snap/pycharm-professional/278',
                       'SNAP_USER_COMMON': '/home/auvsl/snap/pycharm-professional/common',
                       'PYTHONPATH': '/home/auvsl/catkin_ws/devel/lib/python2.7/dist-packages:/opt/ros/melodic/lib/python2.7/dist-packages',
                       'SNAP_COOKIE': 'KVHmahN_WPi0HfHdI7xqUdlL8cEgMH2uTesvW0a3N4TcvWrtL-u1',
                       'SSH_AUTH_SOCK': '/run/user/1000/keyring/ssh', 'SNAP_INSTANCE_KEY': '', 'GDMSESSION': 'ubuntu',
                       'IM_CONFIG_PHASE': '2', 'TEXTDOMAINDIR': '/usr/share/locale/',
                       'GNOME_SHELL_SESSION_MODE': 'ubuntu', 'SNAP_NAME': 'pycharm-professional',
                       'XDG_CONFIG_DIRS': '/etc/xdg/xdg-ubuntu:/etc/xdg', 'SNAP_REEXEC': '',
                       'SNAP_INSTANCE_NAME': 'pycharm-professional',
                       'ROS_ROOT': '/home/auvsl/catkin_ws/src/ros/core/rosbuild', 'XDG_SESSION_ID': '1',
                       'DESKTOP_STARTUP_ID': 'gnome-shell-1589-auvsl-ubuntu18-env-0_TIME100529', '_': '/usr/bin/python',
                       'GTK_MODULES': 'gail:atk-bridge', 'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/run/user/1000/bus',
                       'GTK_IM_MODULE': 'ibus', 'DESKTOP_SESSION': 'ubuntu',
                       'SNAP_COMMON': '/var/snap/pycharm-professional/common', 'LESSCLOSE': '/usr/bin/lesspipe %s %s',
                       'DEFAULTS_PATH': '/usr/share/gconf/ubuntu.default.path',
                       'ROSLISP_PACKAGE_DIRECTORIES': '/home/auvsl/catkin_ws/devel/share/common-lisp', 'SHLVL': '1',
                       'PWD': '/home/auvsl/python3_ws/src/anfis_rl', 'ROS_ETC_DIR': '/opt/ros/melodic/etc/ros',
                       'ROS_MASTER_URI': 'http://localhost:11311', 'ROS_VERSION': '1',
                       'SNAP': '/snap/pycharm-professional/278', 'XDG_MENU_PREFIX': 'gnome-',
                       'LS_COLORS': 'rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=01;31:*.taz=01;31:*.lha=01;31:*.lz4=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.tzo=01;31:*.t7z=01;31:*.zip=01;31:*.z=01;31:*.Z=01;31:*.dz=01;31:*.gz=01;31:*.lrz=01;31:*.lz=01;31:*.lzo=01;31:*.xz=01;31:*.zst=01;31:*.tzst=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.alz=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.cab=01;31:*.wim=01;31:*.swm=01;31:*.dwm=01;31:*.esd=01;31:*.jpg=01;35:*.jpeg=01;35:*.mjpg=01;35:*.mjpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=00;36:*.au=00;36:*.flac=00;36:*.m4a=00;36:*.mid=00;36:*.midi=00;36:*.mka=00;36:*.mp3=00;36:*.mpc=00;36:*.ogg=00;36:*.ra=00;36:*.wav=00;36:*.oga=00;36:*.opus=00;36:*.spx=00;36:*.xspf=00;36:',
                       'GJS_DEBUG_TOPICS': 'JS ERROR;JS LOG', 'XDG_SEAT': 'seat0'}

            self._roslaunch = subprocess.Popen(command, executable='/bin/bash', env=environ)
        else:
            self._roslaunch = subprocess.Popen(
                [sys.executable, os.path.join(ros_path, b"roslaunch"), "-p", self.port, fullpath])
        print("Gazebo launched!")

        self.gzclient_pid = 0

        # Launch the simulation with the given launchfile name
        rospy.init_node('gym', anonymous=True)

        ################################################################################################################
        r = rospy.Rate(1)
        self.clock_sub = rospy.Subscriber('/clock', Clock, self.callback, queue_size=1000000)
        while not rospy.is_shutdown():
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
            r.sleep()
        ################################################################################################################

    def callback(self, message):
        """
        Callback method for the subscriber of the clock topic
        :param message:
        :return:
        """
        # self.last_clock_msg = int(str(message.clock.secs) + str(message.clock.nsecs)) / 1e6
        # print("Message", message)
        self.last_clock_msg = message
        # print("Message", message)

    def step(self, action):

        # Implement this method in every subclass
        # Perform a step in gazebo. E.g. move the robot
        raise NotImplementedError

    def reset(self):

        # Implemented in subclass
        raise NotImplementedError

    def _render(self, mode="human", close=False):

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

    def _close(self):

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

        if (gzclient_count or gzserver_count or roscore_count or rosmaster_count > 0):
            os.wait()

    def _configure(self):

        # TODO
        # From OpenAI API: Provides runtime configuration to the enviroment
        # Maybe set the Real Time Factor?
        pass

    def _seed(self):

        # TODO
        # From OpenAI API: Sets the seed for this env's random number generator(s)
        pass


if __name__ == '__main__':
    a = random.randint(0, 10000)

    GazeboEnv("/home/auvsl/catkin_ws/src/anfis_rl/multi_simulation.sh", ros_path=r"/opt/ros/melodic/bin/roscore",
              arguments=(str(a),))
