#!/bin/bash

for range in 0.05-0.30 0.05-0.35 0.05-0.40 0.10-0.30 0.10-0.35 0.10-0.40; do
  python -m run_sample \
    --cam_out_dir ../resim/results/pipeline/resnet50v2/irnet/make_cam/seg-model-224/cam_outputs \
    --ir_label_out_dir ../resim/results/pipeline/resnet50v2/irnet/ir_label-${range}/seg-model-224/ir_label_outputs \
    --sem_seg_out_dir result/${range}/sem_seg \
    --ins_seg_out_dir result/${range}/ins_seg \
    --irn_weights_name sess/${range}/res50_irn.pth \
    --log_name resnet50v2_${range} \
    --backbone resnet50
done
