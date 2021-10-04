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


## Activate the environment:

run: `source ../../py3venv/bin/activate`

do another source: `source ~/python3_ws/devel/setup.bash`

roslaunch jackal_gazebo empty_world.launch joystick:=false gui:=false

catkin clean; catkin build --cmake-args -DPYTHON_EXECUTABLE:FILEPATH=/home/auvsl/python3_ws/py3env/bin/python

##Tensorboard run
## Local machine
`ssh -L 16007:127.0.0.1:16007 nvidia@192.168.0.12`

## Server
`tensorboard --logdir=runs --samples_per_plugin images=999 --port=16007`
tensorboard --logdir=runs --samples_per_plugin images=999

also 



git clone https://github.com/ros/ros_comm.git
pip install defusedxml netifaces

# option 1

pip install -U catkin_tools

# option 2

git clone https://github.com/catkin/catkin_tools.git
cd catkin_tools

pip3 install -r requirements.txt --upgrade
python3 setup.py install --record install_manifest.txt
python3 setup.py develop

[comment]: <> (python3 setup.py develop)

git clone https://github.com/ros/roscpp_core

run simulation.sh for best results
