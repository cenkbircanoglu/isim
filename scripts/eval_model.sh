#!/bin/bash
#
# Brief description of your script
# Copyright 2021 33755

for model_name in seg-model-4 seg-model-49 seg-model-99 seg-model-149 seg-model-199 final-model; do
    python -m make_cam \
        backbone=resnet50v2 \
        weights=$(pwd)/results/train-unet/train-seg-unet/resnet50/2021-11-04/23-49-59/weights/${model_name}.pt \
        hydra.run.dir=$(pwd)/results/train-unet/train-seg-unet/resnet50/2021-11-04/23-49-59/${model_name}/make_cam \
        infer_list=$(pwd)/voc12/train.txt

    python -m eval_cam \
        cam_out_dir=$(pwd)/results/train-unet/train-seg-unet/resnet50/2021-11-04/23-49-59/${model_name}/make_cam/cam_outputs \
        hydra.run.dir=$(pwd)/results/train-unet/train-seg-unet/resnet50/2021-11-04/23-49-59/${model_name}/eval_cam \
        infer_set=train
done
