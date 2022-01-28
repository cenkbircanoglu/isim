#!/bin/bash

run_experiment() {
    BACKBONE=$1
    CROP_SIZE=$2
    BATCH_SIZE=$3
    CRF_FREQ=$4

    python -m train \
        backbone=${BACKBONE} \
        train_list=$(pwd)/voc12/train_aug.txt \
        val_list=$(pwd)/voc12/train.txt \
        crop_size=${CROP_SIZE} \
        batch_size=${BATCH_SIZE} \
        hydra.run.dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/train \
        crf_freq=${CRF_FREQ} \
        epochs=1000
}

# run_experiment resnet50v2 512 48 2
# run_experiment resnet50v2 512 48 5
# run_experiment resnet50v2 512 48 10
# run_experiment resnet50v2 512 48 25
# run_experiment resnet50v2 512 48 50

run_experiment resnet101v2 512 32 25
# run_experiment resnet152v2 512 32 25
# run_experiment resnest50 512 48 25
# run_experiment resnest101 512 32 25
# run_experiment resnest200 512 16 25
# run_experiment resnest269 512 16 25

