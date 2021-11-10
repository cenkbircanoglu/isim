#!/bin/bash

for CROP_SIZE in 256 320 448 512; do
    for CAM_EVAL_THRESHOLD in 0.1 0.15 0.2 0.25 0.3; do
        for BACKBONE in resnet50v1 resnet50v2 resnet50v3 resnet50v4; do
            for model_name in seg-model-4 seg-model-24 final-model; do
                python -m eval_cam \
                    cam_out_dir=$(pwd)/results/pipeline/${BACKBONE}/crop${CROP_SIZE}/eval/make_cam/${model_name}/cam_outputs \
                    hydra.run.dir=$(pwd)/results/pipeline/${BACKBONE}/crop${CROP_SIZE}/eval/eval_cam/${model_name}_${CAM_EVAL_THRESHOLD} \
                    infer_set=train \
                    logs=$(pwd)/results/pipeline/${BACKBONE}/crop${CROP_SIZE}/eval/logs_${CAM_EVAL_THRESHOLD} \
                    last_epoch=50 \
                    cam_eval_thres=${CAM_EVAL_THRESHOLD}
            done
        done
    done
done


# BEST resnet50v2 0.15 512
