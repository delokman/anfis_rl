#!/bin/bash

echo "PORT:"
read PORT
tensorboard --logdir=~/catkin_ws/src/sam/anfis_rl2/runs --samples_per_plugin images=999 --port=$PORT
