#!/bin/bash
#
# Brief description of your script
# Copyright 2021 33755
backbone=resnet50v1
crop=256

for model_name in seg-model-4 seg-model-49 seg-model-99 seg-model-149 seg-model-199 final-model; do
    python -m make_cam \
        backbone=resnet50v2 \
        weights=$(pwd)/results/pipeline/${backbone}/crop${crop}/train/weights/${model_name}.pt \
        hydra.run.dir=$(pwd)/results/pipeline/${backbone}/crop${crop}/train/make_cam \
        infer_list=$(pwd)/voc12/train.txt

    python -m eval_cam \
        cam_out_dir=$(pwd)/results/pipeline/${backbone}/crop${crop}/train/make_cam/cam_outputs \
        hydra.run.dir=$(pwd)/results/pipeline/${backbone}/crop${crop}/train/eval_cam \
        infer_set=train
done
