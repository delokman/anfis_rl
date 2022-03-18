#!/bin/bash

echo "PORT:"
read PORT
#tensorboard --logdir=~/catkin_ws/src/anfis_rl/runs --samples_per_plugin images=999 --port=$PORT
tensorboard --logdir=~/python3_ws/src/anfis_rl/runs --samples_per_plugin images=999 --port=$PORT
