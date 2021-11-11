#!/bin/bash

run_experiment() {

    CAM_EVAL_THRES=$1

    BACKBONE=resnet50v2
    python -m train \
        backbone=${BACKBONE} \
        train_list=$(pwd)/voc12/train.txt \
        val_list=$(pwd)/voc12/val.txt \
        crop_size=512 \
        batch_size=24 \
        hydra.run.dir=$(pwd)/results/threshold/${BACKBONE}/cam_eval_thres${CAM_EVAL_THRES}/train \
        crf_freq=25 \
        epochs=50 \
        cam_eval_thres=${CAM_EVAL_THRES}

}

run_experiment 0.1
run_experiment 0.15
run_experiment 0.2
run_experiment 0.25
run_experiment 0.3
run_experiment 0.35
run_experiment 0.4
run_experiment 0.45
run_experiment 0.5
run_experiment 0.55
run_experiment 0.6
