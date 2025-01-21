#!/bin/bash


# <valid loss 낮은 epoch의 weight으로 prediction>

ckpt_DFHiC="/data/HiHiC/checkpoints/checkpoints_DFHiC/00023_0.05.15_0.0009907218.npz"
ckpt_DeepHiC="/data/HiHiC/checkpoints/checkpoints_DeepHiC/00011_0.15.29_0.0020183752"
ckpt_HiCARN1="/data/HiHiC/checkpoints/checkpoints_HiCARN1/00019_0.10.48_0.0014813891"
ckpt_HiCARN2="/data/HiHiC/checkpoints/checkpoints_HiCARN2/00013_0.12.33_0.0014848912"
ckpt_HiCNN2="/data/HiHiC/checkpoints/checkpoints_HiCNN/00052_1.03.50_0.0015621280"
ckpt_SRHiC="/data/HiHiC/checkpoints/checkpoints_SRHiC/00227_0.28.08_0.0011104286-240768.meta"
ckpt_iEnhance="/data/HiHiC/checkpoints/checkpoints_iEnhance/00192_6 days, 4.16.06_0.0000305396"
ckpt_HiCPlus="/data/HiHiC/checkpoints/checkpoints_HiCPlus/10000_3.21.07_0.0050615421"

dir_output="./data_model_out"
models=("iEnhance") # "HiCARN1" "HiCARN2" "HiCNN2" "DeepHiC")  # tensorflow: "SRHiC" "DFHiC" / torch: "HiCARN1" "HiCARN2" "HiCNN2" "DeepHiC" "iEnhance" "HiCPlus"
flags=("trial1" "trial2" "trial3" "trial4" "trial5")

for model in "${models[@]}"; do
    ckpt_var="ckpt_${model}"    
    checkpoint="${!ckpt_var}"  # Dynamic variable reference
    
    if [ ${model} == "DFHiC" ] || [ "${model}" == "DeepHiC" ] || [ "${model}" == "HiCPlus" ] || [ ${model} == "SRHiC" ]; then
        dir_input=($(find "./data_model/data_${model}/ENHANCEMENT" -type f -name "*.npz"))
        for input in "${dir_input[@]}"; do
            bash model_prediction.sh -m ${model} -c "${checkpoint}" -b 16 -g "0" -i "${input}" -o "${dir_output}"
        done
    elif [ "${model}" == "HiCARN1" ] || [ "${model}" == "HiCARN2" ]; then # 모델 1&2
        dir_input=($(find "./data_model/data_HiCARN/ENHANCEMENT" -type f -name "*.npz"))
        for input in "${dir_input[@]}"; do
            bash model_prediction.sh -m ${model} -c "${checkpoint}" -b 16 -g "0" -i "${input}" -o "${dir_output}"
        done
    elif [ ${model} == "HiCNN2" ]; then # 모델명과 데이터 폴더명 차이
        dir_input=($(find "./data_model/data_HiCNN/ENHANCEMENT" -type f -name "*.npz"))
        for input in "${dir_input[@]}"; do
            bash model_prediction.sh -m ${model} -c "${checkpoint}" -b 16 -g "0" -i "${input}" -o "${dir_output}"
        done
    elif [ ${model} == "iEnhance" ]; then
        dir_input=($(find "./data_model/data_${model}/ENHANCEMENT" -type f -name "*.npz"))
        for input in "${dir_input[@]}"; do

        base_name=$(basename "${input}")  # 파일 이름만 가져오기 (경로 제거)
        file_name="${base_name%.*}"       # 확장자 제거
        output_file="./data_model_out/OUTPUT__/${file_name}_iEnhance_00192ep.npz"

            if [[ -e "${output_file}" ]]; then
                echo "Skipping: ${output_file} already exists."
                continue
            fi            
            bash model_prediction.sh -m "${model}" -c "${checkpoint}" -b 16 -g "0" -i "${input}" -o "${dir_output}"
        done
    fi 
done

