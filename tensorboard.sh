#!/bin/bash

echo "PORT:"
read PORT
tensorboard --logdir=runs --samples_per_plugin images=999 --port=$PORT
