#!/bin/bash

i=1
for user in "$@"; do
  echo "Namespace - $i: $user"
  i=$((i + 1))
done

echo "Setting URDF Extra to="
export JACKAL_URDF_EXTRAS=$HOME'/python3_ws/src/anfis_rl/urdf/friction.urdf'
echo "Set JACKAL_URDF_EXTRAS=$JACKAL_URDF_EXTRAS"
# exit gracefully by returning a status

source $HOME/catkin_ws/devel/setup.bash

roscd anfis_rl
cp $HOME'/python3_ws/src/anfis_rl/launch/multi_simulation.launch' $HOME'/catkin_ws/src/anfis_rl/launch/multi_simulation.launch'
cp $HOME'/python3_ws/src/anfis_rl/multi_simulation.sh' $HOME'/catkin_ws/src/anfis_rl/multi_simulation.sh'

echo "roslaunch anfis_rl multi_simulation.launch -p "$1" "ns:=$2"  "createWorld:=$3" "x:=$4" "y:=$5" "gui:=$6""
/usr/bin/python2 /opt/ros/melodic/bin/roslaunch $HOME'/catkin_ws/src/anfis_rl/launch/multi_simulation.launch' -p "$1" "createWorld:=$2" "ns:=$3" "x:=$4" "y:=$5" "gui:=$6"
exit 0
