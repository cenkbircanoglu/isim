#python -m train backbone=resnet50v1 epochs=16 crop_size=256 batch_size=96
#python -m train backbone=resnet50v1 epochs=16 crop_size=320 batch_size=64
#python -m train backbone=resnet50v1 epochs=16 crop_size=448 batch_size=64
#python -m train backbone=resnet50v1 epochs=16 crop_size=512 batch_size=32
#
#python -m train backbone=resnet50v2 epochs=16 crop_size=256 batch_size=96
#python -m train backbone=resnet50v2 epochs=16 crop_size=320 batch_size=64
#python -m train backbone=resnet50v2 epochs=16 crop_size=448 batch_size=64
#python -m train backbone=resnet50v2 epochs=16 crop_size=512 batch_size=24
#
#python -m train backbone=resnet50v3 epochs=16 crop_size=256 batch_size=64
#python -m train backbone=resnet50v3 epochs=16 crop_size=320 batch_size=32
#python -m train backbone=resnet50v3 epochs=16 crop_size=448 batch_size=32
#python -m train backbone=resnet50v3 epochs=16 crop_size=512 batch_size=16
#
#python -m train backbone=resnet50v4 epochs=16 crop_size=256 batch_size=8
#python -m train backbone=resnet50v4 epochs=16 crop_size=320 batch_size=8
#python -m train backbone=resnet50v4 epochs=16 crop_size=448 batch_size=8
#python -m train backbone=resnet50v4 epochs=16 crop_size=512 batch_size=4

#python -m train backbone=resnet50v1 train_list=$(pwd)/voc12/train.txt val_list=$(pwd)/voc12/val.txt crop_size=256 batch_size=96 hydra.run.dir=$(pwd)/results/pipeline/resnet50v1/crop256/train crf_freq=25 epochs=50
#python -m train backbone=resnet50v2 train_list=$(pwd)/voc12/train.txt val_list=$(pwd)/voc12/val.txt crop_size=256 batch_size=96 hydra.run.dir=$(pwd)/results/pipeline/resnet50v2/crop256/train crf_freq=25 epochs=50
#python -m train backbone=resnet50v3 train_list=$(pwd)/voc12/train.txt val_list=$(pwd)/voc12/val.txt crop_size=256 batch_size=64 hydra.run.dir=$(pwd)/results/pipeline/resnet50v3/crop256/train crf_freq=25 epochs=50
#python -m train backbone=resnet50v4 train_list=$(pwd)/voc12/train.txt val_list=$(pwd)/voc12/val.txt crop_size=256 batch_size=8 hydra.run.dir=$(pwd)/results/pipeline/resnet50v4/crop256/train crf_freq=25 epochs=50
#
#python -m train backbone=resnet50v1 train_list=$(pwd)/voc12/train.txt val_list=$(pwd)/voc12/val.txt crop_size=320 batch_size=64 hydra.run.dir=$(pwd)/results/pipeline/resnet50v1/crop320/train crf_freq=25 epochs=50
#python -m train backbone=resnet50v2 train_list=$(pwd)/voc12/train.txt val_list=$(pwd)/voc12/val.txt crop_size=320 batch_size=64 hydra.run.dir=$(pwd)/results/pipeline/resnet50v2/crop320/train crf_freq=25 epochs=50
#python -m train backbone=resnet50v3 train_list=$(pwd)/voc12/train.txt val_list=$(pwd)/voc12/val.txt crop_size=320 batch_size=32 hydra.run.dir=$(pwd)/results/pipeline/resnet50v3/crop320/train crf_freq=25 epochs=50
#python -m train backbone=resnet50v4 train_list=$(pwd)/voc12/train.txt val_list=$(pwd)/voc12/val.txt crop_size=320 batch_size=8 hydra.run.dir=$(pwd)/results/pipeline/resnet50v4/crop320/train crf_freq=25 epochs=50

python -m train backbone=resnet50v1 train_list=$(pwd)/voc12/train.txt val_list=$(pwd)/voc12/val.txt crop_size=448 batch_size=32 hydra.run.dir=$(pwd)/results/pipeline/resnet50v1/crop448/train crf_freq=25 epochs=50
python -m train backbone=resnet50v2 train_list=$(pwd)/voc12/train.txt val_list=$(pwd)/voc12/val.txt crop_size=448 batch_size=32 hydra.run.dir=$(pwd)/results/pipeline/resnet50v2/crop448/train crf_freq=25 epochs=50
python -m train backbone=resnet50v3 train_list=$(pwd)/voc12/train.txt val_list=$(pwd)/voc12/val.txt crop_size=448 batch_size=16 hydra.run.dir=$(pwd)/results/pipeline/resnet50v3/crop448/train crf_freq=25 epochs=50
python -m train backbone=resnet50v4 train_list=$(pwd)/voc12/train.txt val_list=$(pwd)/voc12/val.txt crop_size=448 batch_size=6 hydra.run.dir=$(pwd)/results/pipeline/resnet50v4/crop448/train crf_freq=25 epochs=50

python -m train backbone=resnet50v1 train_list=$(pwd)/voc12/train.txt val_list=$(pwd)/voc12/val.txt crop_size=512 batch_size=32 hydra.run.dir=$(pwd)/results/pipeline/resnet50v1/crop512/train crf_freq=25 epochs=50
python -m train backbone=resnet50v2 train_list=$(pwd)/voc12/train.txt val_list=$(pwd)/voc12/val.txt crop_size=512 batch_size=24 hydra.run.dir=$(pwd)/results/pipeline/resnet50v2/crop512/train crf_freq=25 epochs=50
python -m train backbone=resnet50v3 train_list=$(pwd)/voc12/train.txt val_list=$(pwd)/voc12/val.txt crop_size=512 batch_size=12 hydra.run.dir=$(pwd)/results/pipeline/resnet50v3/crop512/train crf_freq=25 epochs=50
python -m train backbone=resnet50v4 train_list=$(pwd)/voc12/train.txt val_list=$(pwd)/voc12/val.txt crop_size=512 batch_size=4 hydra.run.dir=$(pwd)/results/pipeline/resnet50v4/crop512/train crf_freq=25 epochs=50


#
#python -m make_cam backbone=resnet50v2 weights=$(pwd)/results/train-unet/train-seg-unet/resnet50/2021-11-04/23-49-59/weights/final-model.pt hydra.run.dir=$(pwd)/results/train-unet/train-seg-unet/resnet50/2021-11-04/23-49-59/make_cam
#python -m eval_cam cam_out_dir=$(pwd)/results/train-unet/train-seg-unet/resnet50/2021-11-04/23-49-59/make_cam/cam_outputs hydra.run.dir=$(pwd)/results/train-unet/train-seg-unet/resnet50/2021-11-04/23-49-59/eval_cam
#
#python -m make_cam backbone=resnet50v2 weights=$(pwd)/results/train-unet/train-seg-unet/resnet50/2021-11-04/23-49-59/weights/final-model.pt hydra.run.dir=$(pwd)/results/train-unet/train-seg-unet/resnet50/2021-11-04/23-49-59/make_cam infer_list=$(pwd)/voc12/train.txt
#python -m eval_cam cam_out_dir=$(pwd)/results/train-unet/train-seg-unet/resnet50/2021-11-04/23-49-59/make_cam/cam_outputs hydra.run.dir=$(pwd)/results/train-unet/train-seg-unet/resnet50/2021-11-04/23-49-59/eval_cam infer_set=train
#
#
#
#python -m make_cam backbone=resnet50v2 weights=$(pwd)/results/train-unet/train-seg-unet/resnet50/2021-11-04/23-49-59/weights/seg-model-4.pt hydra.run.dir=$(pwd)/results/train-unet/train-seg-unet/resnet50/2021-11-04/23-49-59/model-4/make_cam infer_list=$(pwd)/voc12/train.txt
#python -m eval_cam cam_out_dir=$(pwd)/results/train-unet/train-seg-unet/resnet50/2021-11-04/23-49-59/model-4make_cam/cam_outputs hydra.run.dir=$(pwd)/results/train-unet/train-seg-unet/resnet50/2021-11-04/23-49-59/model-4/eval_cam infer_set=train
