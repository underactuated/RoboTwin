#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_negative_data_failure_detection.py \
  "$task_name" "$task_config" \
  --episode-num 1 \
  --perturbation-amplitude 0.25 \
  --replay-shift-growth-factor 2.0 \
  --replay-shift-max-amplitude 8.0
