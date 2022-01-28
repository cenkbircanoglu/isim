#!/bin/bash

BACKBONE=resnet50v2
CRF_FREQ=2
CROP_SIZE=512
INFER_SET=train

# for CAM_EVAL_THRESHOLD in 0.1 0.15 0.2 0.25; do
#     python -m eval_cam \
#         cam_out_dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/make_cam/seg-model- \
#         hydra.run.dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/eval_cam/seg-model-${model_name}_${CAM_EVAL_THRESHOLD} \
#         infer_set=$INFER_SET \
#         logs=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/logs_${CAM_EVAL_THRESHOLD} \
#         epochs=\"4,5,7,9,11,13,15,17,19,21,final-model\" \
#         cam_eval_thres=${CAM_EVAL_THRESHOLD} \
#         crf_freq=$CRF_FREQ \
#         crop_size=$CROP_SIZE \
#         backbone=$BACKBONE
# done
#
# CRF_FREQ=5
#
# for CAM_EVAL_THRESHOLD in 0.1 0.15 0.2 0.25; do
#     python -m eval_cam \
#         cam_out_dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/make_cam/seg-model- \
#         hydra.run.dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/eval_cam/seg-model-${model_name}_${CAM_EVAL_THRESHOLD} \
#         infer_set=$INFER_SET \
#         logs=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/logs_${CAM_EVAL_THRESHOLD} \
#         epochs=\"4,9,14,19,24,29,34,39,44,49,final-model\" \
#         cam_eval_thres=${CAM_EVAL_THRESHOLD} \
#         crf_freq=$CRF_FREQ \
#         crop_size=$CROP_SIZE \
#         backbone=$BACKBONE
# done
#
#
# CRF_FREQ=10
#
# for CAM_EVAL_THRESHOLD in 0.1 0.15 0.2 0.25; do
#     python -m eval_cam \
#         cam_out_dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/make_cam/seg-model- \
#         hydra.run.dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/eval_cam/seg-model-${model_name}_${CAM_EVAL_THRESHOLD} \
#         infer_set=$INFER_SET \
#         logs=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/logs_${CAM_EVAL_THRESHOLD} \
#         epochs=\"4,9,19,29,39,49,59,69,79,89,final-model\" \
#         cam_eval_thres=${CAM_EVAL_THRESHOLD} \
#         crf_freq=$CRF_FREQ \
#         crop_size=$CROP_SIZE \
#         backbone=$BACKBONE
# done
#
#
# CRF_FREQ=25
#
# for CAM_EVAL_THRESHOLD in 0.1 0.15 0.2 0.25; do
#     python -m eval_cam \
#         cam_out_dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/make_cam/seg-model- \
#         hydra.run.dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/eval_cam/seg-model-${model_name}_${CAM_EVAL_THRESHOLD} \
#         infer_set=$INFER_SET \
#         logs=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/logs_${CAM_EVAL_THRESHOLD} \
#         epochs=\"4,24,49,74,99,124,149,174,199,224,final-model\" \
#         cam_eval_thres=${CAM_EVAL_THRESHOLD} \
#         crf_freq=$CRF_FREQ \
#         crop_size=$CROP_SIZE \
#         backbone=$BACKBONE
# done


# CRF_FREQ=50
#
# for CAM_EVAL_THRESHOLD in 0.1 0.15 0.2 0.25; do
#     python -m eval_cam \
#         cam_out_dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/make_cam/seg-model- \
#         hydra.run.dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/eval_cam/seg-model-${model_name}_${CAM_EVAL_THRESHOLD} \
#         infer_set=$INFER_SET \
#         logs=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/logs_${CAM_EVAL_THRESHOLD} \
#         epochs=\"4,49,99,149,199,249,299,349,399,449,final-model\" \
#         cam_eval_thres=${CAM_EVAL_THRESHOLD} \
#         crf_freq=$CRF_FREQ \
#         crop_size=$CROP_SIZE \
#         backbone=$BACKBONE
# done


BACKBONE=resnet101v2
CRF_FREQ=25

for CAM_EVAL_THRESHOLD in 0.1 0.15 0.2 0.25; do
    python -m eval_cam \
        cam_out_dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/make_cam/seg-model- \
        hydra.run.dir=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/eval_cam/seg-model-${model_name}_${CAM_EVAL_THRESHOLD} \
        infer_set=$INFER_SET \
        logs=$(pwd)/outputs/pipeline/${BACKBONE}-crf_freq-${CRF_FREQ}-crop_size-${CROP_SIZE}/eval/logs_${CAM_EVAL_THRESHOLD} \
        epochs=\"4,24,49,74,99,124,149,174,199,224,final-model\" \
        cam_eval_thres=${CAM_EVAL_THRESHOLD} \
        crf_freq=$CRF_FREQ \
        crop_size=$CROP_SIZE \
        backbone=$BACKBONE
done
