#!/bin/bash

BACKBONE=resnet50v2

for TR_CAM_EVAL_THRES in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6; do
    for CAM_EVAL_THRESHOLD in 0.1 0.15 0.2 0.25 0.3; do
        python -m eval_cam \
            cam_out_dir=$(pwd)/results/threshold/${BACKBONE}/cam_eval_thres${TR_CAM_EVAL_THRES}/eval/make_cam/cam_outputs \
            hydra.run.dir=$(pwd)/results/threshold/${BACKBONE}/cam_eval_thres${TR_CAM_EVAL_THRES}/eval/eval_cam/${CAM_EVAL_THRESHOLD} \
            infer_set=train \
            logs=$(pwd)/results/threshold/${BACKBONE}/cam_eval_thres${TR_CAM_EVAL_THRES}/eval/logs_${CAM_EVAL_THRESHOLD} \
            last_epoch=50 \
            cam_eval_thres=${CAM_EVAL_THRESHOLD}
    done
done
