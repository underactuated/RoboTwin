#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_negative_data.py $task_name $task_config \
--negative-mode replay_object_shift \
--perturbation-amplitude 0.25 \
--replay-shift-growth-factor 2.0 \
--replay-shift-max-amplitude 8.0 \
--save-setting "$task_config"
