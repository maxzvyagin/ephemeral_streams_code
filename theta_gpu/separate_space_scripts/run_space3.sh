#!/bin/bash
singularity shell --nv -B /lus:/lus /lus/theta-fs0/software/thetagpu/nvidia-containers/tensorflow2/tf2_20.10-py3.simg
cd ~/ephemeral_streams_code/theta_gpu/separate_space_scripts
python ~/ephemeral_streams_code/theta_gpu/theta_running.py -m run -s 4
python ~/ephemeral_streams_code/theta_gpu/theta_running.py -m run -s 5