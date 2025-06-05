#!/bin/bash

# <valid loss 낮은 epoch의 weight으로 prediction>
# ckpt_DFHiC="/data/HiHiC/checkpoints_3.0M/checkpoints_DFHiC/00023_0.05.15_0.0009907218.npz"
# ckpt_DeepHiC="/data/HiHiC/checkpoints_3.0M/checkpoints_DeepHiC/00011_0.15.29_0.0020183752"
# ckpt_HiCARN1="/data/HiHiC/checkpoints_3.0M/checkpoints_HiCARN1/00019_0.10.48_0.0014813891"
# ckpt_HiCARN2="/data/HiHiC/checkpoints_3.0M/checkpoints_HiCARN2/00013_0.12.33_0.0014848912"
# ckpt_HiCNN2="/data/HiHiC/checkpoints_3.0M/checkpoints_HiCNN/00052_1.03.50_0.0015621280"
# ckpt_SRHiC="/data/HiHiC/checkpoints_3.0M/checkpoints_SRHiC/00227_0.28.08_0.0011104286-240768.meta"
# ckpt_iEnhance="/data/HiHiC/checkpoints_3.0M/checkpoints_iEnhance/00192_6 days, 4.16.06_0.0000305396"
# ckpt_HiCPlus="/data/HiHiC/checkpoints_3.0M/checkpoints_HiCPlus_10000ep/10000_3.21.07_0.0050615421"

ckpt_DFHiC="/data/HiHiC/checkpoints_10.0M/checkpoints_DFHiC/00024_0.10.39_0.0006255576.npz"
ckpt_DeepHiC="/data/HiHiC/checkpoints_10.0M/checkpoints_DeepHiC/00104_3.46.21_0.0011968986"
ckpt_HiCARN1="/data/HiHiC/checkpoints_10.0M/checkpoints_HiCARN1/00008_0.09.13_0.0009015936"
ckpt_HiCARN2="/data/HiHiC/checkpoints_10.0M/checkpoints_HiCARN2/00010_0.19.04_0.0009090670"
ckpt_HiCNN2="/data/HiHiC/checkpoints_10.0M/checkpoints_HiCNN/00021_0.51.09_0.0009432364"
ckpt_SRHiC="/data/HiHiC/checkpoints_10.0M/checkpoints_SRHiC/00238_0.54.40_0.0007267343-496881.meta"
ckpt_iEnhance="/data/HiHiC/checkpoints_10.0M/checkpoints_iEnhance_1000ep/00251_6 days, 8.52.27_0.0000370489"
ckpt_HiCPlus="/data/HiHiC/checkpoints_10.0M/checkpoints_HiCPlus_10000ep/10000_6.30.59_0.0021362852"

dir_output="./data_model_out"
models=("SRHiC" "DFHiC")  # tensorflow: "SRHiC" "DFHiC" / torch: "HiCARN1" "HiCARN2" "HiCNN2" "DeepHiC" "iEnhance" "HiCPlus"
# flags=("trial1" "trial2" "trial3" "trial4" "trial5")

for model in "${models[@]}"; do
    ckpt_var="ckpt_${model}"    
    checkpoint="${!ckpt_var}"  # Dynamic variable reference
    
    if [ ${model} == "DFHiC" ] || [ "${model}" == "DeepHiC" ] || [ "${model}" == "HiCPlus" ] || [ ${model} == "SRHiC" ]; then
        dir_input=($(find "./data_model/data_${model}/TEST" -type f -name "*.npz"))
        for input in "${dir_input[@]}"; do
            bash model_prediction.sh -m ${model} -c "${checkpoint}" -b 16 -g "0" -i "${input}" -o "${dir_output}"
        done
    elif [ "${model}" == "HiCARN1" ] || [ "${model}" == "HiCARN2" ]; then # 모델 1&2
        dir_input=($(find "./data_model/data_HiCARN/TEST" -type f -name "*.npz"))
        for input in "${dir_input[@]}"; do
            bash model_prediction.sh -m ${model} -c "${checkpoint}" -b 16 -g "0" -i "${input}" -o "${dir_output}"
        done
    elif [ ${model} == "HiCNN2" ]; then # 모델명과 데이터 폴더명 차이
        dir_input=($(find "./data_model/data_HiCNN/TEST" -type f -name "*.npz"))
        for input in "${dir_input[@]}"; do
            bash model_prediction.sh -m ${model} -c "${checkpoint}" -b 16 -g "0" -i "${input}" -o "${dir_output}"
        done
    elif [ ${model} == "iEnhance" ]; then
        dir_input=($(find "./data_model/data_${model}/TEST" -type f -name "*.npz"))
        for input in "${dir_input[@]}"; do        
            bash model_prediction.sh -m "${model}" -c "${checkpoint}" -b 16 -g "0" -i "${input}" -o "${dir_output}"
        done
    fi 
done
