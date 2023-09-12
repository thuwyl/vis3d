#!/bin/bash

source /home/wyl/anaconda3/bin/activate /home/wyl/anaconda3/envs/open3d
WORKSPACE="/home/wyl/ws/vis3d" 
cd ${WORKSPACE}
export PYTHONPATH=${PYTHONPATH}:${WORKSPACE}

python tools/vis_kitti.py --index 1