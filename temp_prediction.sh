#!/bin/bash

# <valid loss 낮은 epoch의 weight으로 prediction>

ckpt_DFHiC="./checkpoints_DFHiC/00024_0.10.39_0.0006255576.npz"
ckpt_DeepHiC="./checkpoints_DeepHiC/00104_3.46.21_0.0011968986"
ckpt_HiCARN1="./checkpoints_HiCARN1/00008_0.09.13_0.0009015936"
ckpt_HiCARN2="./checkpoints_HiCARN2/00010_0.19.04_0.0009090670"
ckpt_HiCNN2="./checkpoints_HiCNN/00021_0.51.09_0.0009432364"
ckpt_SRHiC="./checkpoints_SRHiC/00238_0.54.40_0.0007267343-496881.meta"
ckpt_iEnhance="./checkpoints_iEnhance/00282_6 days, 8.56.43_0.0000446402"
ckpt_HiCPlus="./checkpoints_HiCPlus/01000_0.39.06_0.0041882079"

dir_output="./data_model_out"
models=("SRHiC" "DFHiC" "HiCARN1" "HiCARN2" "HiCNN2" "DeepHiC" "HiCPlus" "iEnhance")  # tensorflow: "SRHiC" "DFHiC" / torch: "HiCARN1" "HiCARN2" "HiCNN2" "DeepHiC" "iEnhance" "HiCPlus"
flags=("trial1" "trial2" "trial3" "trial4" "trial5")

for model in "${models[@]}"; do
    ckpt_var="ckpt_${model}"    
    checkpoint="${!ckpt_var}"  # Dynamic variable reference
    
    if [ ${model} == "DFHiC" ] || [ "${model}" == "DeepHiC" ] || [ "${model}" == "HiCPlus" ]; then
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
    elif [ ${model} == "iEnhance" ]; then # data_make_whole.py 안함
        dir_input=($(find "./data_model/data_${model}/enhance" -type f -name "*.npz"))
        for input in "${dir_input[@]}"; do
            prefix=${input%%__*}
            bash model_prediction.sh -m ${model} -c "${checkpoint}" -b 16 -g "-1" -i "${input}" -o "${dir_output}/${prefix}"
        done
    fi 
done

# <prediction output submatrix를 intra chromosome 으로 통합>
dir=($(find "${dir_output}/OUTPUT" -type f -name "*.npz"))
for flag in "${flags[@]}"; do
    mkdir -p "${dir_output}/GM12878_${flag}"
    mkdir -p "${dir_output}/K562_${flag}"
    mkdir -p "${dir_output}/CH12-LX_${flag}"
done

for file in "${dir}"/*; do
    model=$(echo "$file" | awk -F'_' 'print $5')
    # iEnhance가 아닌 경우에만 실행
    if [[ ${model} != "iEnhance" ]]; then
        python data_make_whole.py -i "${file}" -o "${dir_output}/${file%%__*}"
    fi
done
