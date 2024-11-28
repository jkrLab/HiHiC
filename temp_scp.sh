#!/bin/bash

scp -r ./targeted_data/ mohyelim7@203.253.2.168:/home/data/HiHiC
scp -r ./targeted_data_downsample/ mohyelim7@203.253.2.168:/home/data/HiHiC

models=("SRHiC" "DFHiC" "HiCARN1" "HiCARN2" "HiCNN2" "DeepHiC" "HiCPlus", "iEnhance")

for model in "${models[@]}"; do
    scp ./predicted_${model}/* mohyelim7@203.253.2.168:/home/data/HiHiC/predicted_by_${model}
done