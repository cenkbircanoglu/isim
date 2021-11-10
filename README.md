

Python version 3.7


# Train
```bash
python -m train
```


```bash
python -m make_cam \
  encoder=${ENCODER} \
  decoder=${DECODER} \
  weights=$OUTPUTDIR/weights/cls-model.pt \
  hydra.run.dir=$OUTPUTDIR/make_cam_cls \
  output_dir=$OUTPUTDIR/make_cam_cls/cams \
  infer_list=$(pwd)/voc12/train.txt

for i in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4; do
  python -m eval_cam \
    --config-name=eval_cam \
    cam_out_dir=$OUTPUTDIR/make_cam_cls/cams \
    infer_set=train \
    hydra.run.dir=$OUTPUTDIR/eval_cam_cls \
    name=${EXPERIMENT}-cls-cam-${i} \
    tag=$TAG \
    project=$PROJECT \
    cam_eval_thres=${i}
done

# EVALUATE CAM Unet
python -m make_cam \
  encoder=${ENCODER} \
  decoder=${DECODER} \
  weights=$OUTPUTDIR/weights/final-model.pt \
  hydra.run.dir=$OUTPUTDIR/make_cam \
  output_dir=$OUTPUTDIR/make_cam/cams \
  infer_list=$(pwd)/voc12/train.txt
```
