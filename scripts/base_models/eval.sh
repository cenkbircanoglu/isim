#!/bin/bash

run() {
    BACKBONE=$1
    CROP_SIZE=$2
    MODEL_NAME=$3
    CAM_EVAL_THRESHOLD=$4

    python -m eval_cam \
        cam_out_dir=$(pwd)/results/base_models/${BACKBONE}/crop${CROP_SIZE}/eval/make_cam/${MODEL_NAME}/cam_outputs \
        hydra.run.dir=$(pwd)/results/base_models/${BACKBONE}/crop${CROP_SIZE}/eval/eval_cam/${MODEL_NAME}_${CAM_EVAL_THRESHOLD} \
        infer_set=train \
        logs=$(pwd)/results/base_models/${BACKBONE}/crop${CROP_SIZE}/eval/logs_${CAM_EVAL_THRESHOLD} \
        last_epoch=50 \
        cam_eval_thres=${CAM_EVAL_THRESHOLD}
}

run resnet50v2 512 seg-model-4 0.15
run resnet50v2 512 seg-model-24 0.15
#run resnet50v2 512 final-model 0.15
