#!/bin/bash

run_experiment() {
    BACKBONE=$1
    CROP_SIZE=$2
    BATCH_SIZE=$3
    CRF_FREQ=25
    EPOCHS=1000

    python -m train \
        backbone=${BACKBONE} \
        train_list=$(pwd)/voc12/train.txt \
        val_list=$(pwd)/voc12/val.txt \
        crop_size=${CROP_SIZE} \
        batch_size=${BATCH_SIZE} \
        hydra.run.dir=$(pwd)/results/pipeline/${BACKBONE}/crop${CROP_SIZE}/train \
        crf_freq=${CRF_FREQ} \
        epochs=${EPOCHS}
}

run_experiment resnet50v2 512 24
