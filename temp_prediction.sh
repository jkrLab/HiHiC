#!/bin/bash

# <valid loss 낮은 epoch의 weight으로 prediction>

ckpt_DFHiC="./checkpoints_DFHiC/00024_0.10.39_0.0006255576.npz"
ckpt_DeepHiC="./checkpoints_DeepHiC/00104_3.46.21_0.0011968986"
ckpt_HiCARN1="./checkpoints_HiCARN1/00008_0.09.13_0.0009015936"
ckpt_HiCARN2="./checkpoints_HiCARN2/00010_0.19.04_0.0009090670"
ckpt_HiCNN2="./checkpoints_HiCNN/00021_0.51.09_0.0009432364"
ckpt_SRHiC="./checkpoints_SRHiC/00238_0.54.40_0.0007267343-496881.meta"
ckpt_iEnhance="./checkpoints_iEnhance/00282_6 days, 8.56.43_0.0000446402"
ckpt_HiCPlus="./checkpoints_HiCPlus/10000_6.30.59_0.0021362852"

dir_output="./data_model_out"
models=("iEnhance")  # tensorflow: "SRHiC" "DFHiC" / torch: "HiCARN1" "HiCARN2" "HiCNN2" "DeepHiC" "iEnhance" "HiCPlus"
flags=("trial1" "trial2" "trial3" "trial4" "trial5")

for model in "${models[@]}"; do
    ckpt_var="ckpt_${model}"    
    checkpoint="${!ckpt_var}"  # Dynamic variable reference
    
    if [ ${model} == "DFHiC" ] || [ "${model}" == "DeepHiC" ] || [ "${model}" == "HiCPlus" ] || [ ${model} == "iEnhance" ]; then
        dir_input=($(find "./data_model/data_${model}/enhance" -type f -name "*.npz"))
        for input in "${dir_input[@]}"; do
            bash model_prediction.sh -m ${model} -c "${checkpoint}" -b 16 -g "-1" -i "${input}" -o "${dir_output}"
        done
    elif [ ${model} == "SRHiC" ]; then # .npy
        dir_input=($(find "./data_model/data_SRHiC/enhance" -type f -name "*.npy"))
        for input in "${dir_input[@]}"; do
            bash model_prediction.sh -m ${model} -c "${checkpoint}" -b 16 -g "-1" -i "${input}" -o "${dir_output}"
        done
    elif [ "${model}" == "HiCARN1" ] || [ "${model}" == "HiCARN2" ]; then # 모델 1&2
        dir_input=($(find "./data_model/data_HiCARN/enhance" -type f -name "*.npz"))
        for input in "${dir_input[@]}"; do
            bash model_prediction.sh -m ${model} -c "${checkpoint}" -b 16 -g "-1" -i "${input}" -o "${dir_output}"
        done
    elif [ ${model} == "HiCNN2" ]; then # 모델명과 데이터 폴더명 차이
        dir_input=($(find "./data_model/data_HiCNN/enhance" -type f -name "*.npz"))
        for input in "${dir_input[@]}"; do
            bash model_prediction.sh -m ${model} -c "${checkpoint}" -b 16 -g "-1" -i "${input}" -o "${dir_output}"
        done
    fi 
done

