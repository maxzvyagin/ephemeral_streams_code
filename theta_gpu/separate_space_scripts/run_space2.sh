#!/bin/bash
singularity shell --nv -B /lus:/lus /lus/theta-fs0/software/thetagpu/nvidia-containers/tensorflow2/tf2_20.10-py3.simg
cd ~/hyper_resilient/theta_gpu/separate_space_scripts
python ~/hyper_resilient/theta_gpu/theta_running.py -m run -s 2
python ~/hyper_resilient/theta_gpu/theta_running.py -m run -s 3