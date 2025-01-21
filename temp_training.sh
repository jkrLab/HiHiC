#!/bin/bash
epoch=500
batch=16
dir_output="./checkpoints"
dir_log="./log"

models=("DeepHiC" "HiCPlus" "HiCNN" "iEnhance")  # tensorflow: "SRHiC" "DFHiC" / torch: "HiCARN1" "HiCARN2" "HiCNN2" "DeepHiC" "iEnhance" "HiCPlus"
for model in "${models[@]}"; do
    bash model_train.sh -m "${model}" -e "${epoch}" -b "${batch}" -g "0" -o "${dir_output}" -l "${dir_log}" -t "./data_model/data_${model}/TRAIN" -v "./data_model/data_${model}/VALID"
done

models=("HiCARN1" "HiCARN2")  # tensorflow: "SRHiC" "DFHiC" / torch: "HiCARN1" "HiCARN2" "HiCNN2" "DeepHiC" "iEnhance" "HiCPlus"
for model in "${models[@]}"; do
    bash model_train.sh -m "${model}" -e "${epoch}" -b "${batch}" -g "0" -o "${dir_output}" -l "${dir_log}" -t "./data_model/data_HiCARN/TRAIN" -v "./data_model/data_HiCARN/VALID"
done

models=("SRHiC" "DFHiC")  # tensorflow: "SRHiC" "DFHiC" / torch: "HiCARN1" "HiCARN2" "HiCNN2" "DeepHiC" "iEnhance" "HiCPlus"
for model in "${models[@]}"; do
    bash model_train.sh -m "${model}" -e "${epoch}" -b "${batch}" -g "0" -o "${dir_output}" -l "${dir_log}" -t "./data_model/data_${model}/TRAIN" -v "./data_model/data_${model}/VALID"
done

models=("HiCPlus")  # tensorflow: "SRHiC" "DFHiC" / torch: "HiCARN1" "HiCARN2" "HiCNN2" "DeepHiC" "iEnhance" "HiCPlus"
for model in "${models[@]}"; do
    bash model_train.sh -m "${model}" -e "10000" -b "${batch}" -g "0" -o "${dir_output}" -l "${dir_log}" -t "./data_model/data_${model}/TRAIN" -v "./data_model/data_${model}/VALID"
done
