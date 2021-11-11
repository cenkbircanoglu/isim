#!/bin/bash

run_experiment() {
    BACKBONE=$1
    CROP_SIZE=$2
    BATCH_SIZE=$3
    CRF_FREQ=$4
    EPOCHS=$5

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
