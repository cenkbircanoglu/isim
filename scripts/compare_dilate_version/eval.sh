#!/bin/bash

BACKBONE=resnet50v2

for DILATE_VERSION in 1 2; do
    for CAM_EVAL_THRESHOLD in 0.1 0.15 0.2 0.25 0.3; do
        python -m eval_cam \
            cam_out_dir=$(pwd)/results/dilation/${BACKBONE}/dilate_version${DILATE_VERSION}/eval/make_cam/cam_outputs \
            hydra.run.dir=$(pwd)/results/dilation/${BACKBONE}/dilate_version${DILATE_VERSION}/eval/eval_cam/${CAM_EVAL_THRESHOLD} \
            infer_set=train \
            logs=$(pwd)/results/dilation/${BACKBONE}/dilate_version${DILATE_VERSION}/eval/logs_${CAM_EVAL_THRESHOLD} \
            last_epoch=50 \
            cam_eval_thres=${CAM_EVAL_THRESHOLD}
    done
done
