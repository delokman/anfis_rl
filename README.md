# ROS ANFIS RL

<p align="center">
    <a href="https://github.com/AUVSL/anfis_rl/graphs/contributors" alt="Contributors">
        <img src="https://img.shields.io/github/contributors/AUVSL/anfis_rl" /></a>
    <a href="https://github.com/AUVSL/anfis_rl/pulse" alt="Activity">
        <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/AUVSL/anfis_rl"></a>
    <a href="https://github.com/AUVSL/anfis_rl/stargazers">
        <img alt="Stars" src="https://img.shields.io/github/stars/AUVSL/anfis_rl"></a>
    <a href="https://github.com/AUVSL/anfis_rl/network/members">
        <img alt="Forks" src="https://img.shields.io/github/forks/AUVSL/anfis_rl"></a>
    <a href="https://github.com/AUVSL/anfis_rl/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/AUVSL/anfis_rl"></a>
    <a href="./LICENSE" alt="Activity">
        <img alt="GitHub" src="https://img.shields.io/github/license/AUVSL/anfis_rl"></a>
</p>

## Setup

Clone the repository into the Python 3 catkin workspace

### Simulation

1. In a non python 3 environment. Source the `anfis` repository
2. Run `./simulation.sh` script to start Gazebo environment

### Program

1. Source the `anfis` repository in Python 3 environment
   1. source the Python 3 virtual environment as stated below in section `Python 3 Setup` using `souce /path/to/py3venv/bin/active`
   2. Source the ROS catkin workspace `source ~/python3_ws/devel/setup.bash`
2. Ros run or run in Python 3 virtual environment `main.py`

## Activate the environment:

run: `source ../../py3venv/bin/activate`

do another source: `source ~/python3_ws/devel/setup.bash`

## Tensorboard log visualizations
### Local machine
`ssh -L 16007:127.0.0.1:16007 nvidia@192.168.0.12`

### Server
`tensorboard --logdir=runs --samples_per_plugin images=999 --port=16007`

`tensorboard --logdir=runs --samples_per_plugin images=999`

Otherwise, run the script: `tensorboard.sh` in there select the port to set up the server to.

## Python 3 environment setup for ROS Melodic

The tutorial is based on https://youtu.be/oxK4ykVh1EE?t=1232 

The instructions steps are also demonstrated here. There are some differences demonstrated below.
https://github.com/AUVSL/Jackal-maintenance-and-upgrade/blob/main/python3_setup_instuctions.sh

Line 28 be sure to change the file path to the location of your python virtual environment.
Rebuilding the Python environment
`catkin clean; catkin build --cmake-args -DPYTHON_EXECUTABLE:FILEPATH=/home/auvsl/python3_ws/py3env/bin/python`

### Ros launch setup

`git clone https://github.com/ros/ros_comm.git`

`pip install defusedxml netifaces`

### option 1 to try and get roslaunch and rosrun to run in Python3 Melodic

`pip install -U catkin_tools`

### option 2 to try and get roslaunch and rosrun to run in Python3 Melodic

`git clone https://github.com/catkin/catkin_tools.git`

`cd catkin_tools`

`pip3 install -r requirements.txt --upgrade`

`python3 setup.py install --record install_manifest.txt`

`python3 setup.py develop`

[comment]: <> (python3 setup.py develop)

`git clone https://github.com/ros/roscpp_core`
