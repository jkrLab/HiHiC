#!/bin/bash
epoch=500
batch=16
dir_output="./checkpoints"
dir_log="./log"

models=("SRHiC" "DFHiC" "HiCARN1" "HiCARN2" "HiCNN2" "DeepHiC" "iEnhance" "HiCPlus")  # tensorflow: "SRHiC" "DFHiC" / torch: "HiCARN1" "HiCARN2" "HiCNN2" "DeepHiC" "iEnhance" "HiCPlus"
for model in "${models[@]}"; do
    bash model_train.sh -m "${model}" -e "${epoch}" -b "${batch}" -g "-1" -o "${dir_output}" -l "${dir_log}" -t "./data_model/data_${model}/TRAIN" -v "./data_model/data_${model}/VALID"
done
