#!/bin/bash

echo "Setting URDF Extra to="
export JACKAL_URDF_EXTRAS=$HOME'/python3_ws/src/anfis_rl/urdf/friction.urdf'
echo "Set JACKAL_URDF_EXTRAS=$JACKAL_URDF_EXTRAS"
# exit gracefully by returning a status

source $HOME/catkin_ws/devel/setup.bash

roscd anfis_rl
git pull

roslaunch anfis_rl simulation.launch
exit 0

