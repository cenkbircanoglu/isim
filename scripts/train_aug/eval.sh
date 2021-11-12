#!/bin/bash

BACKBONE=resnet50v2
CAM_EVAL_THRESHOLD=0.15

for model_name in 4 199 224 249 274 299 324 349 374 399 424 449 474 499 524 549 574; do
    python -m eval_cam \
        cam_out_dir=$(pwd)/results/pipeline/${BACKBONE}/eval/make_cam/seg-model-${model_name}/cam_outputs \
        hydra.run.dir=$(pwd)/results/pipeline/${BACKBONE}/eval/eval_cam/seg-model-${model_name}_${CAM_EVAL_THRESHOLD} \
        infer_set=train \
        logs=$(pwd)/results/pipeline/${BACKBONE}/eval/logs_${CAM_EVAL_THRESHOLD} \
        last_epoch=300 \
        cam_eval_thres=${CAM_EVAL_THRESHOLD}
done
