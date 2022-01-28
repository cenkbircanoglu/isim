#!/bin/bash

make_cam_experiment() {

    BACKBONE=$1
    CROP_SIZE=$2
    CRF_FREQ=$3

    python -m irnet.make_cam_hr \
        backbone=${BACKBONE} \
        weights=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/train/weights/final-model.pth \
        hydra.run.dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/make_cam/ \
        train_list=$(pwd)/voc12/train_aug.txt \
        crop_size=${CROP_SIZE}
}

make_cam_experiment resnet50v2 512 25
make_cam_experiment resnet101v2 512 25

