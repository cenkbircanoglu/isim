#!/bin/bash
BACKBONE=resnet50v2
model_name=224

conf_fg_thres=0.30
conf_bg_thres=0.05

python -m irnet.cam_to_ir_label \
    hydra.run.dir=$(pwd)/results/pipeline/${BACKBONE}/irnet/ir_label-${conf_bg_thres}-${conf_fg_thres}/seg-model-${model_name} \
    train_list=$(pwd)/voc12/train_aug.txt \
    cam_out_dir=$(pwd)/results/pipeline/${BACKBONE}/irnet/make_cam/seg-model-${model_name}/cam_outputs \
    conf_fg_thres=${conf_fg_thres} \
    conf_bg_thres=${conf_bg_thres}


conf_fg_thres=0.35
conf_bg_thres=0.05

python -m irnet.cam_to_ir_label \
    hydra.run.dir=$(pwd)/results/pipeline/${BACKBONE}/irnet/ir_label-${conf_bg_thres}-${conf_fg_thres}/seg-model-${model_name} \
    train_list=$(pwd)/voc12/train_aug.txt \
    cam_out_dir=$(pwd)/results/pipeline/${BACKBONE}/irnet/make_cam/seg-model-${model_name}/cam_outputs \
    conf_fg_thres=${conf_fg_thres} \
    conf_bg_thres=${conf_bg_thres}


conf_fg_thres=0.30
conf_bg_thres=0.10

python -m irnet.cam_to_ir_label \
    hydra.run.dir=$(pwd)/results/pipeline/${BACKBONE}/irnet/ir_label-${conf_bg_thres}-${conf_fg_thres}/seg-model-${model_name} \
    train_list=$(pwd)/voc12/train_aug.txt \
    cam_out_dir=$(pwd)/results/pipeline/${BACKBONE}/irnet/make_cam/seg-model-${model_name}/cam_outputs \
    conf_fg_thres=${conf_fg_thres} \
    conf_bg_thres=${conf_bg_thres}

conf_fg_thres=0.35
conf_bg_thres=0.10

python -m irnet.cam_to_ir_label \
    hydra.run.dir=$(pwd)/results/pipeline/${BACKBONE}/irnet/ir_label-${conf_bg_thres}-${conf_fg_thres}/seg-model-${model_name} \
    train_list=$(pwd)/voc12/train_aug.txt \
    cam_out_dir=$(pwd)/results/pipeline/${BACKBONE}/irnet/make_cam/seg-model-${model_name}/cam_outputs \
    conf_fg_thres=${conf_fg_thres} \
    conf_bg_thres=${conf_bg_thres}

conf_fg_thres=0.40
conf_bg_thres=0.10

python -m irnet.cam_to_ir_label \
    hydra.run.dir=$(pwd)/results/pipeline/${BACKBONE}/irnet/ir_label-${conf_bg_thres}-${conf_fg_thres}/seg-model-${model_name} \
    train_list=$(pwd)/voc12/train_aug.txt \
    cam_out_dir=$(pwd)/results/pipeline/${BACKBONE}/irnet/make_cam/seg-model-${model_name}/cam_outputs \
    conf_fg_thres=${conf_fg_thres} \
    conf_bg_thres=${conf_bg_thres}

conf_fg_thres=0.40
conf_bg_thres=0.05

python -m irnet.cam_to_ir_label \
    hydra.run.dir=$(pwd)/results/pipeline/${BACKBONE}/irnet/ir_label-${conf_bg_thres}-${conf_fg_thres}/seg-model-${model_name} \
    train_list=$(pwd)/voc12/train_aug.txt \
    cam_out_dir=$(pwd)/results/pipeline/${BACKBONE}/irnet/make_cam/seg-model-${model_name}/cam_outputs \
    conf_fg_thres=${conf_fg_thres} \
    conf_bg_thres=${conf_bg_thres}
