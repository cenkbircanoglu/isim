#!/bin/bash

source $(dirname $0)/train.sh

run_experiment resnet50v2 512 24 25 1000
