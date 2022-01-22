#!/bin/bash

#run() {
#    BACKBONE=$1
#    CROP_SIZE=$2
#    MODEL_NAME=$3
#    python -m make_cam \
#        backbone=${BACKBONE} \
#        weights=$(pwd)/results/base_models/${BACKBONE}/crop${CROP_SIZE}/train/weights/${MODEL_NAME}.pt \
#        hydra.run.dir=$(pwd)/results/base_models/${BACKBONE}/crop${CROP_SIZE}/eval/make_cam/${MODEL_NAME} \
#        infer_list=$(pwd)/voc12/train.txt \
#        crop_size=${CROP_SIZE}
#}
#
#run resnet50v2 512 seg-model-4
#run resnet50v2 512 seg-model-24
##run resnet50v2 512 final-model


python -m make_cam \
        backbone=resnet50v2 \
        weights=$(pwd)/results/train-pipeline/resnet50v2/2021-11-22/23-26-38/weights/best-model-20.pt \
        hydra.run.dir=$(pwd)/results/train-pipeline/resnet50v2/2021-11-22/23-26-38/make_cam/ \
        infer_list=$(pwd)/voc12/train.txt \
        crop_size=512


python -m eval_cam \
    cam_out_dir=$(pwd)/results/train-pipeline/resnet50v2/2021-11-22/23-26-38/make_cam/cam_outputs \
    hydra.run.dir=$(pwd)/results/train-pipeline/resnet50v2/2021-11-22/23-26-38/eval_cam \
    infer_set=train \
    logs=$(pwd)/results/train-pipeline/resnet50v2/2021-11-22/23-26-38/logs \
    last_epoch=50 \
    cam_eval_thres=0.15
