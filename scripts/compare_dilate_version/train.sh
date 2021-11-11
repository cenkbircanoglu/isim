#!/bin/bash

run_experiment() {

    DILATE_VERSION=$1
    BACKBONE=resnet50v2
    python -m train \
        backbone=${BACKBONE} \
        train_list=$(pwd)/voc12/train.txt \
        val_list=$(pwd)/voc12/val.txt \
        crop_size=512 \
        batch_size=24 \
        hydra.run.dir=$(pwd)/results/dilation/${BACKBONE}/dilate_version${DILATE_VERSION}/train \
        crf_freq=25 \
        epochs=50 \
        dilate_version=${DILATE_VERSION}

}

run_experiment 1
run_experiment 2
