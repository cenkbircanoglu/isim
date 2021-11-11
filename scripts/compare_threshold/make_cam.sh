#!/bin/bash
BACKBONE=resnet50v2
for CAM_EVAL_THRES in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6; do
    python -m make_cam \
        backbone=${BACKBONE} \
        weights=$(pwd)/results/threshold/${BACKBONE}/cam_eval_thres${CAM_EVAL_THRES}/train/weights/final-model.pt \
        hydra.run.dir=$(pwd)/results/threshold/${BACKBONE}/cam_eval_thres${CAM_EVAL_THRES}/eval/make_cam/ \
        infer_list=$(pwd)/voc12/train.txt \
        crop_size=512
done
