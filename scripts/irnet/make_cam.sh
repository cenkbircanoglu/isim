#!/bin/bash
BACKBONE=resnet50v2
CROP_SIZE=512
model_name=224

python -m irnet.make_cam_hr \
    backbone=${BACKBONE} \
    weights=$(pwd)/results/pipeline/${BACKBONE}/train/weights/seg-model-${model_name}.pt \
    hydra.run.dir=$(pwd)/results/pipeline/${BACKBONE}/irnet/make_cam/seg-model-${model_name} \
    train_list=$(pwd)/voc12/train_aug.txt \
    crop_size=${CROP_SIZE}


BACKBONE=resnet101v2
CROP_SIZE=512
model_name=249

python -m irnet.make_cam_hr \
    backbone=${BACKBONE} \
    weights=$(pwd)/results/pipeline/${BACKBONE}/train/weights/seg-model-${model_name}.pt \
    hydra.run.dir=$(pwd)/results/pipeline/${BACKBONE}/irnet/make_cam/seg-model-${model_name} \
    train_list=$(pwd)/voc12/train_aug.txt \
    crop_size=${CROP_SIZE}
