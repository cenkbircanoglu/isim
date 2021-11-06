#!/bin/bash

backbone=resnet50v2
crop=320

for model_name in seg-model-4 seg-model-19 seg-model-39 final-model; do
    python -m make_cam \
        backbone=${backbone} \
        weights=$(pwd)/results/pipeline/${backbone}/crop${crop}/train/weights/${model_name}.pt \
        hydra.run.dir=$(pwd)/results/pipeline/${backbone}/crop${crop}/eval/make_cam/${model_name} \
        infer_list=$(pwd)/voc12/train.txt

    python -m eval_cam \
        cam_out_dir=$(pwd)/results/pipeline/${backbone}/crop${crop}/eval/make_cam/${model_name}/cam_outputs \
        hydra.run.dir=$(pwd)/results/pipeline/${backbone}/crop${crop}/eval/eval_cam/${model_name} \
        infer_set=train \
        logs=$(pwd)/results/pipeline/${backbone}/crop${crop}/eval/logs \
        last_epoch=50
done
