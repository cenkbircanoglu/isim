#!/bin/bash

for CROP_SIZE in 256 320 448 512; do
    for BACKBONE in resnet50v1 resnet50v2 resnet50v3 resnet50v4; do
        for model_name in seg-model-4 seg-model-24 final-model; do
            python -m make_cam \
                backbone=${BACKBONE} \
                weights=$(pwd)/results/pipeline/${BACKBONE}/crop${CROP_SIZE}/train/weights/${model_name}.pt \
                hydra.run.dir=$(pwd)/results/pipeline/${BACKBONE}/crop${CROP_SIZE}/eval/make_cam/${model_name} \
                infer_list=$(pwd)/voc12/train.txt \
                crop_size=${CROP_SIZE}
        done
    done
done
