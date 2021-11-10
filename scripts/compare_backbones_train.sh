#!/bin/bash

source $(dirname $0)/train.sh

run_experiment resnet50v1 256 96 25 50
run_experiment resnet50v2 256 96 25 50
run_experiment resnet50v3 256 64 25 50
run_experiment resnet50v4 256 8 25 50

run_experiment resnet50v1 320 64 25 50
run_experiment resnet50v2 320 64 25 50
run_experiment resnet50v3 320 32 25 50
run_experiment resnet50v4 320 8 25 50

run_experiment resnet50v1 448 32 25 50
run_experiment resnet50v2 448 32 25 50
run_experiment resnet50v3 448 16 25 50
run_experiment resnet50v4 448 6 25 50

run_experiment resnet50v1 512 32 25 50
run_experiment resnet50v2 512 24 25 50
run_experiment resnet50v3 512 12 25 50
run_experiment resnet50v4 512 4 25 50
