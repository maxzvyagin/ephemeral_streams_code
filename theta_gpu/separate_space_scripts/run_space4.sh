#!/bin/bash
#singularity shell --nv -B /lus:/lus /lus/theta-fs0/software/thetagpu/nvidia-containers/tensorflow2/tf2_20.10-py3.simg
#cd ~/ephemeral_streams_code/theta_gpu
python ~/ephemeral_streams_code/theta_gpu/theta_running.py -m run -s 6 -o ten_trials.csv -t 10
python ~/ephemeral_streams_code/theta_gpu/theta_running.py -m run -s 7 -o ten_trials.csv -t 10