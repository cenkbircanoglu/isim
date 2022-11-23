
# ISIM
The official implementation of "ISIM: Iterative Self-Improved Model for Weakly Supervised Segmentation".

## Citation

- In Review for [IJCV](https://www.springer.com/journal/11263/)
- Please cite our paper if the code is helpful to your research. [arxiv](https://arxiv.org/abs/2211.12455)


## Overview
![Overall architecture](./resources/pipeline.png)

<br>

# Prerequisite
- Python 3.7.11, PyTorch 1.7.0, and more in requirements.txt
- pydensecrf
- CUDA 11.5
- RTX 3090 GPU, RTX 2080 GPU (x4)

# Usage

## Install python dependencies
```bash
pip install -r requirements.txt
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

## Download PASCAL VOC 2012 devkit
Follow instructions in http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit

## 1. Run the experiments
```bash
./scripts/experiments.sh
```

## 2. Get prediction results for the trained models
```bash
./scripts/infer_experiments.sh
```

## 3. Evaluate the predictions
```bash
./scripts/eval_experiments.sh
```

## 5. Results

| Methods | val mIoU | test mIoU |
|---|---|-------:|
| ISIM with ResNet-101 | [70.51](http://host.robots.ox.ac.uk:8080/anonymous/EGU15D.html) | [71.45](http://host.robots.ox.ac.uk:8080/anonymous/YG2SXH.html) |
| ISIM with ResNeSt-200 |  [74.90](http://host.robots.ox.ac.uk:8080/anonymous/BYTTBW.html) | [74.98](http://host.robots.ox.ac.uk:8080/anonymous/XUU3KG.html) |

Qualitative segmentation results on the PASCAL VOC 2012 validation set.

![Qualitative Results](./resources/cams.png)

## 6. Provide the trained weights and training logs

- Release the final masks by our models. (SOON)

| Model                  | val | test |
|:----------------------:|:---:|:----:|
| DeepLabv3+ ResNet-101 | [val.tgz]() | [test.tgz]() |
| DeepLabv3+ ResNeSt-200 | [val.tgz]() | [test.tgz]() |

<br>

For any issues, please contact <b>Cenk Bircanoglu</b>, cenk.bircanoglu@gmail.com

## 7. Notes

- In MID-2021, Project started. 
- 15 Nov 2021, [paper](./resources/VISI-D-21-00725.pdf) submitted to [IJCV](https://www.springer.com/journal/11263/).
- On 21 January 2022, Got a refusal from [IJCV](https://www.springer.com/journal/11263/). 
- On 25 January 2022, Cenk Bircanoglu got the Covid-19 Vaccine and had severe health problems.
- On 20 November 2022, Health issues were relatively possible to manage, and we decided to continue this project.
- On 21 November 2022, Found out [RecurSeed and EdgePredictMix](https://arxiv.org/abs/2204.06754v3). A similar idea (honestly, with a lot of improvements) has already been published.
- On 23 November 2022, Decided to put the paper to [arxiv](https://arxiv.org/abs/2211.12455), share the code, note the project in history, and start from scratch with a new idea.
