#!/bin/bash

run_experiment() {
    BACKBONE=$1
    CROP_SIZE=$2
    BATCH_SIZE=$3
    CRF_FREQ=25
    EPOCHS=50

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

run_experiment resnet50v1 256 96
run_experiment resnet50v2 256 96
run_experiment resnet50v3 256 64
run_experiment resnet50v4 256 8

run_experiment resnet50v1 320 64
run_experiment resnet50v2 320 64
run_experiment resnet50v3 320 32
run_experiment resnet50v4 320 8

run_experiment resnet50v1 448 32
run_experiment resnet50v2 448 32
run_experiment resnet50v3 448 16
run_experiment resnet50v4 448 6

run_experiment resnet50v1 512 32
run_experiment resnet50v2 512 24
run_experiment resnet50v3 512 12
run_experiment resnet50v4 512 4
