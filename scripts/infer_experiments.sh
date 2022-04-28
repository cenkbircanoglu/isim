#!/bin/bash

infer_experiment() {
    BACKBONE=$1
    CROP_SIZE=$2
    CRF_FREQ=$3
    local -n arr=$4

    for model_name in ${arr[@]}; do
        python -m make_cam \
            backbone=${BACKBONE} \
            weights=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/train/weights/seg-model-${model_name}.pth \
            hydra.run.dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/make_cam/seg-model-${model_name} \
            infer_list=$(pwd)/voc12/train.txt \
            crop_size=${CROP_SIZE}

        python -m make_cam \
            backbone=${BACKBONE} \
            weights=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/train/weights/seg-model-${model_name}.pth \
            hydra.run.dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/make_cam/seg-model-${model_name} \
            infer_list=$(pwd)/voc12/val.txt \
            crop_size=${CROP_SIZE}
    done

    python -m make_cam \
        backbone=${BACKBONE} \
        weights=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/train/weights/final-model.pth \
        hydra.run.dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/make_cam/seg-model-final-model \
        infer_list=$(pwd)/voc12/train.txt \
        crop_size=${CROP_SIZE}

    python -m make_cam \
        backbone=${BACKBONE} \
        weights=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/train/weights/final-model.pth \
        hydra.run.dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/make_cam/seg-model-final-model \
        infer_list=$(pwd)/voc12/val.txt \
        crop_size=${CROP_SIZE}
}

# Experiments on comparing the different iteration frequency

# CRF_FREQ = 2
array=(4 5 7 9 11 13 15 17 19 21)
infer_experiment resnet50v2 512 2 array

# CRF_FREQ = 5
array=(4 9 14 19 24 29 34 39 44 49)
infer_experiment resnet50v2 512 5 array

# CRF_FREQ = 10
array=(4 9 19 29 39 49 59 69 79 89)
infer_experiment resnet50v2 512 10 array

# CRF_FREQ = 25
array=(4 24 49 74 99 124 149 174 199 224)
infer_experiment resnet50v2 512 25 array

# CRF_FREQ = 50
array=(4 49 99 149 199 249 299 349 399 449)
infer_experiment resnet50v2 512 50 array

# Experiments on comparing the different network architectures

# CRF_FREQ = 25 (as it gives the best mIoU and time-complexity)
array=(4 24 49 74 99 124 149 174 199 224)

# ResNet Variations
infer_experiment resnet50v2 512 25 array
infer_experiment resnet101v2 512 25 array
infer_experiment resnet152v2 512 25 array

# ResNeSt Variations
infer_experiment resnest50 512 25 array
infer_experiment resnest101 512 25 array
infer_experiment resnest200 512 25 array
infer_experiment resnest269 512 25 array
