#!/bin/bash

CROP_SIZE=512
BACKBONE=resnet50v2
model_name=final-model
CAM_EVAL_THRESHOLD=0.15

for SCALES in "1.0" "1.5" "2.0" "3.0" 1.0-0.5 1.0-0.5-1.5 1.0-0.5-1.5-2.0 1.0-2.0 1.0-2.0-1.5 1.0-1.5 1.0-0.5-1.5-2.0-3.0-1.25-1.75 1.0-0.5-1.5-1.25-1.75 1.0-1.5-1.25-1.75; do
    python -m make_cam \
        backbone=${BACKBONE} \
        weights=$(pwd)/results/pipeline/${BACKBONE}/crop${CROP_SIZE}/train/weights/${model_name}.pt \
        hydra.run.dir=$(pwd)/results/pipeline/${BACKBONE}/crop${CROP_SIZE}/${SCALES}/eval/make_cam/${model_name} \
        infer_list=$(pwd)/voc12/train.txt \
        crop_size=${CROP_SIZE} \
        scales=${SCALES}

    python -m eval_cam \
        cam_out_dir=$(pwd)/results/pipeline/${BACKBONE}/crop${CROP_SIZE}/${SCALES}/eval/make_cam/${model_name}/cam_outputs \
        hydra.run.dir=$(pwd)/results/pipeline/${BACKBONE}/crop${CROP_SIZE}/${SCALES}/eval/eval_cam/${model_name}_${CAM_EVAL_THRESHOLD} \
        infer_set=train \
        logs=$(pwd)/results/pipeline/${BACKBONE}/crop${CROP_SIZE}/${SCALES}/eval/logs_${CAM_EVAL_THRESHOLD} \
        last_epoch=50 \
        cam_eval_thres=${CAM_EVAL_THRESHOLD}
done
