#!/bin/bash

BACKBONE=resnet101v2
CAM_EVAL_THRESHOLD=0.15

for model_name in 4 24 49 74 99 124 149 174 199 224 249 274 299 324 349 374 399; do
    python -m eval_cam \
        cam_out_dir=$(pwd)/results/pipeline/${BACKBONE}/eval/make_cam/seg-model-${model_name}/cam_outputs \
        hydra.run.dir=$(pwd)/results/pipeline/${BACKBONE}/eval/eval_cam/seg-model-${model_name}_${CAM_EVAL_THRESHOLD} \
        infer_set=train \
        logs=$(pwd)/results/pipeline/${BACKBONE}/eval/logs_${CAM_EVAL_THRESHOLD} \
        last_epoch=300 \
        cam_eval_thres=${CAM_EVAL_THRESHOLD}
done
