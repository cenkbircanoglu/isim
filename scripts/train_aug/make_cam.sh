#!/bin/bash
BACKBONE=resnet101v2
CROP_SIZE=512

for model_name in 4 24 49 74 99 124 149 174 199 224 249 274 299 324 349 374 399; do
    python -m make_cam \
        backbone=${BACKBONE} \
        weights=$(pwd)/results/pipeline/${BACKBONE}/train/weights/seg-model-${model_name}.pt \
        hydra.run.dir=$(pwd)/results/pipeline/${BACKBONE}/eval/make_cam/seg-model-${model_name} \
        infer_list=$(pwd)/voc12/train.txt \
        crop_size=${CROP_SIZE}
done
