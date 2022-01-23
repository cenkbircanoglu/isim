[![CodeQL](https://github.com/cenkbircanoglu/isim/actions/workflows/codeql-analysis.yml/badge.svg?branch=main)](https://github.com/cenkbircanoglu/isim/actions/workflows/codeql-analysis.yml)

[![pages-build-deployment](https://github.com/cenkbircanoglu/isim/actions/workflows/pages/pages-build-deployment/badge.svg?branch=gh-pages)](https://github.com/cenkbircanoglu/isim/actions/workflows/pages/pages-build-deployment)


[![PWC]()]()
[![PWC]()]()

# ISIM
The official implementation of "ISIM: Iterative Self-Improved Model for Weakly Supervised Segmentation".

## Citation
- In Review for - 
- Please cite our paper if the code is helpful to your research. [arxiv]()
```
```

## Abstract
TODO

## Overview
![Overall architecture]()

<br>

# Prerequisite
- Python 3.7.11, PyTorch 1.7.0, and more in requirements.txt
- pydensecrf
- CUDA 11.5
- RTX 3090 GPU

# Usage

## Install python dependencies
```bash
pip install -r requirements.txt
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

## Download PASCAL VOC 2012 devkit
Follow instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit

## 1. Train an image classifier for generating CAMs
```bash
python -m train_cam hydra.run.dir=results/
```

## 2. Generate CAMs
```bash
python -m cam.make_cam hydra.run.dir=results/ weights=$(pwd)/results/weights/final-model.pt
```

## 3. Evaluate the models
```bash
python -m cam.eval_cam hydra.run.dir=results/ cam_out_dir=$(pwd)/results/cam_outputs/
```

## 5. Results
Qualitative segmentation results on the PASCAL VOC 2012 validation set. 
Top: original images. Middle: ground truth. Bottom: prediction of the segmentation model trained using the pseudo-labels from ISIM.
![Overall architecture]()

| Methods | mIoU |
|---|---:|
| ISIM with ResNet-101 | TODO |
| ISIM with ResNeSt-200 | TODO |

## 6. Provide the trained weights and training logs

- TODO
[experiments.zip]()

- Release the final masks by our models. 

| Model                  | val | test |
|:----------------------:|:---:|:----:|
| DeepLabv3+ ResNet-101 | [val.tgz]() | [test.tgz]() |
| DeepLabv3+ ResNeSt-200 | [val.tgz]() | [test.tgz]() |


### Quick Experiment on PASCAL VOC 2012 with ResNet50 backbone

```bash
sh scripts/run.sh
```

| background | aeroplane | bicycle | bird | boat | bottle | bus | car | cat | chair | cow | diningtable | dog | horse | motorbike | person | pottedplant | sheep | sofa | train | tvmonitor | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8316 | 0.50373 | 0.3227 | 0.5934  | 0.3601 | 0.5890 | 0.6572 | 0.5765 | 0.6845 | 0.2839 | 0.5924 | 0.4507 | 0.5850 | 0.6190 | 0.6776 | 0.6606 | 0.4663 | 0.6561 | 0.5252 | 0.4804 | 0.5352 | 0.5548 |

<br>

For any issues, please contact <b>Cenk Bircanoglu</b>, cenk.bircanoglu@gmail.com

