#!/bin/bash


eval_experiment (){
    BACKBONE=$1
    CRF_FREQ=$2
    CROP_SIZE=$3
    INFER_SET=$4
    EPOCHS=$5

    for CAM_EVAL_THRESHOLD in 0.1 0.15 0.2 0.25; do
        python -m eval_cam \
            cam_out_dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/make_cam/seg-model- \
            hydra.run.dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/eval_cam/seg-model-${model_name}_${CAM_EVAL_THRESHOLD} \
            infer_set=$INFER_SET \
            logs=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/logs_${CAM_EVAL_THRESHOLD} \
            epochs=$EPOCHS \
            cam_eval_thres=${CAM_EVAL_THRESHOLD} \
            crf_freq=$CRF_FREQ \
            crop_size=$CROP_SIZE \
            backbone=$BACKBONE
    done

}
## Experiments on comparing the different iteration frequency

epochs=\"4,5,7,9,11,13,15,17,19,21,final-model\"
eval_experiment resnet50v2 2 512 train $epochs
eval_experiment resnet50v2 2 512 val $epochs

epochs=\"4,9,14,19,24,29,34,39,44,49,final-model\"
eval_experiment resnet50v2 5 512 train $epochs
eval_experiment resnet50v2 5 512 val $epochs

epochs=\"4,9,19,29,39,49,59,69,79,89,final-model\"
eval_experiment resnet50v2 10 512 train $epochs
eval_experiment resnet50v2 10 512 val $epochs

epochs=\"4,24,49,74,99,124,149,174,199,224,final-model\"
eval_experiment resnet50v2 25 512 train $epochs
eval_experiment resnet50v2 25 512 val $epochs

epochs=\"4,49,99,149,199,249,299,349,399,449,final-model\"
eval_experiment resnet50v2 50 512 train $epochs
eval_experiment resnet50v2 50 512 val $epochs


## Experiments on comparing the different network architectures

epochs=\"4,24,49,74,99,124,149,174,199,224,final-model\"
eval_experiment resnet50v2 25 512 train $epochs
eval_experiment resnet50v2 25 512 val $epochs

eval_experiment resnet101v2 25 512 train $epochs
eval_experiment resnet101v2 25 512 val $epochs

eval_experiment resnet152v2 25 512 train $epochs
eval_experiment resnet152v2 25 512 val $epochs

eval_experiment resnest50 25 512 train $epochs
eval_experiment resnest50 25 512 val $epochs

eval_experiment resnest101 25 512 train $epochs
eval_experiment resnest101 25 512 val $epochs

eval_experiment resnest200 25 512 train $epochs
eval_experiment resnest200 25 512 val $epochs

eval_experiment resnest269 25 512 train $epochs
eval_experiment resnest269 25 512 val $epochs
