#!/bin/bash
BACKBONE=resnet50v2
for DILATE_VERSION in 1 2; do
    python -m make_cam \
        backbone=${BACKBONE} \
        weights=$(pwd)/results/dilation/${BACKBONE}/dilate_version${DILATE_VERSION}/train/weights/final-model.pt \
        hydra.run.dir=$(pwd)/results/dilation/${BACKBONE}/dilate_version${DILATE_VERSION}/eval/make_cam/ \
        infer_list=$(pwd)/voc12/train.txt \
        crop_size=512
done
