#!/bin/bash
set -euo pipefail

## step size 50 epoch 마다 예측 수행 ###
seed=42
root_dir=$(pwd)
step_num=50
gpu_id=-1

while getopts ":m:c:b:g:r:i:o:s:" flag; 
do
    case $flag in
        m) model=$OPTARG;;
        c) ckpt_fold=$OPTARG;;
        b) batch_size=$OPTARG;;
        g) gpu_id=$OPTARG;;
        r) read=$OPTARG;; 
        i) input_data=$OPTARG;;
        o) output_data_dir=$OPTARG;;
        s) step_num=$OPTARG;;
        \?)
        echo "Invalid option: -$OPTARG" >&2
        exit 1;;
        :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1;;
    esac
done

# 디렉토리 존재 여부를 확인하는 함수
check_directory_exists() {
    if [ ! -d "$1" ]; then
        echo "Error: The directory '$1' does not exist."
        exit 1
    fi
}

# 파일 존재 여부를 확인하는 함수
check_file_exists() {
    if [ ! -f "$1" ]; then
        echo "Error: The file '$1' does not exist."
        exit 1
    fi
}

# 필수 인수인 --model 및 --ckpt_fold이 제공되었는지 확인합니다.
if [ -z "$model" ] || [ -z "$ckpt_fold" ]; then
    echo "Error: Both --model and --ckpt_fold are required."
    exit 1
fi

check_directory_exists "$ckpt_fold"
check_file_exists "$input_data"

root_dir=$(pwd)

# 저장 디렉토리 존재하지 않는 경우 생성
if [ ! -d "$output_data_dir" ]; then
    mkdir -p "$output_data_dir"
fi

echo ""
echo "  ...Current working directory is ${root_dir}."
echo "      ${model} will enhance low resolution HiC data, ${input_data}"
echo "      And the enhanced HiC data will be saved in ${output_data_dir}." 
echo "      The pretrained model, ${ckpt_fold} will be utilized."
echo "      If available, GPU:${gpu_id} will be used for prediction; otherwise, CPU will be used."
echo ""

if [ "$model" = "HiCARN1" ]; then
    for file in "$ckpt_fold"/*; do
        ckpt=$(basename "$file")
        ckpt_num=${ckpt:0:5}

        if [[ $ckpt_num =~ ^[0-9]+$ ]]; then
            remainder=$((10#$ckpt_num % step_num))

            if [ "$remainder" -eq 0 ]; then
                python model_HiCARN/40x40_Predict.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --read ${read} --input_data ${input_data} --output_data_dir ${output_data_dir}
            fi
        fi
    done

elif [ "$model" = "HiCARN2" ]; then
    for file in "$ckpt_fold"/*; do
        ckpt=$(basename "$file")
        ckpt_num=${ckpt:0:5}

        if [[ $ckpt_num =~ ^[0-9]+$ ]]; then
            remainder=$((10#$ckpt_num % step_num))

            if [ "$remainder" -eq 0 ]; then
                python model_HiCARN/40x40_Predict.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --read ${read} --input_data ${input_data} --output_data_dir ${output_data_dir}
            fi
        fi
    done

elif [ "$model" = "DeepHiC" ]; then
    for file in "$ckpt_fold"/*; do
        ckpt=$(basename "$file")
        ckpt_num=${ckpt:0:5}

        if [[ $ckpt_num =~ ^[0-9]+$ ]]; then
            remainder=$((10#$ckpt_num % step_num))

            if [ "$remainder" -eq 0 ]; then
                python model_DeepHiC/data_predict.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --read ${read} --input_data ${input_data} --output_data_dir ${output_data_dir}
            fi        
        fi
    done

elif [ "$model" = "HiCNN2" ]; then
    for file in "$ckpt_fold"/*; do
        ckpt=$(basename "$file")
        ckpt_num=${ckpt:0:5}

        if [[ $ckpt_num =~ ^[0-9]+$ ]]; then
            remainder=$((10#$ckpt_num % step_num))

            if [ "$remainder" -eq 0 ]; then
                python model_HiCNN2/HiCNN2_predict.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --read ${read} --input_data ${input_data} --output_data_dir ${output_data_dir}
            fi
        fi
    done

elif [ "$model" = "DFHiC" ]; then
    for file in "$ckpt_fold"/*; do
        ckpt=$(basename "$file")
        ckpt_num=${ckpt:0:5}

        if [[ $ckpt_num =~ ^[0-9]+$ ]]; then
            remainder=$((10#$ckpt_num % step_num))

            if [ "$remainder" -eq 0 ]; then
                python model_DFHiC/run_predict.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --read ${read} --input_data ${input_data} --output_data_dir ${output_data_dir}
            fi
        fi
    done

elif [ "$model" = "SRHiC" ]; then
    for file in $(find "$ckpt_fold" -type f -name '*.meta'); do
        ckpt=$(basename "$file")
        ckpt_num=${ckpt:0:5}

        if [[ $ckpt_num =~ ^[0-9]+$ ]]; then
            remainder=$((10#$ckpt_num % step_num))

            if [ "$remainder" -eq 0 ]; then
                python model_SRHiC/src/SRHiC_predict.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --read ${read} --input_data ${input_data} --output_data_dir ${output_data_dir}
            fi
        fi
    done

elif [ "$model" = "HiCPlus" ]; then
    for file in "$ckpt_fold"/*; do
        ckpt=$(basename "$file")
        ckpt_num=${ckpt:0:5}

        if [[ $ckpt_num =~ ^[0-9]+$ ]]; then
            remainder=$((10#$ckpt_num % step_num))

            if [ "$remainder" -eq 0 ]; then
                python model_HiCPlus/HiCPlus/pred_chromosome.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --read ${read} --input_data ${input_data} --output_data_dir ${output_data_dir}
            fi
        fi
    done

elif [ "$model" = "iEnhance" ]; then
    for file in "$ckpt_fold"/*; do
        ckpt=$(basename "$file")
        ckpt_num=${ckpt:0:5}

        if [[ $ckpt_num =~ ^[0-9]+$ ]]; then
            remainder=$((10#$ckpt_num % step_num))

            if [ "$remainder" -eq 0 ]; then
                python model_iEnhance/predict-hic.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --read ${read} --input_data ${input_data} --output_data_dir ${output_data_dir}
            fi
        fi
    done

else
    echo "Model name should be one of the DeepHiC, HiCNN2, DFHiC, HiCPlus, HiCARN1, HiCARN2, SRHiC, iEnhance."
fi


# ### 초반 학습 [1,10] epoch 예측 수행 ###

# seed=42
# root_dir=$(pwd)
# step_num=1
# gpu_id=-1

# # 모델 리스트 정의
# models=("HiCARN" "DeepHiC" "HiCNN" "HiCPlus" "iEnhance" "DFHiC" "SRHiC")

# # 필수 인수 설정
# batch_size=16  # 기본 배치 크기 설정
# down_ratio=16   # 다운 비율 설정

# # 디렉토리 및 파일 존재 여부 확인 함수
# check_directory_exists() {
#     if [ ! -d "$1" ]; then
#         echo "Error: The directory '$1' does not exist."
#         exit 1
#     fi
# }

# check_file_exists() {
#     if [ ! -f "$1" ]; then
#         echo "Error: The file '$1' does not exist."
#         exit 1
#     fi
# }

# echo ""
# echo "  ...Current working directory is ${root_dir}."
# echo "      Models will sequentially enhance low resolution HiC data."
# echo "      GPU:${gpu_id} will be used if available; otherwise, CPU will be used."
# echo ""

# # 모델 목록 순회
# for model in "${models[@]}"; do
#     ckpt_fold="checkpoints_${model}"
#     output_data_dir="output_${model}"

#     if [ "$model" == "SRHiC" ]; then
#         input_data="data_${model}/test_KR_300/test_ratio16.npy"
#     else
#         input_data="data_${model}/test_KR_300/test_ratio16.npz"
#     fi
    
#     # 경로 확인
#     check_directory_exists "$ckpt_fold"
#     check_file_exists "$input_data"
    
#     if [ ! -d "$output_data_dir" ]; then
#         mkdir -p "$output_data_dir"
#     fi

#     echo "Running predictions for model: ${model}, saving outputs to ${output_data_dir}"
    
#     for file in "$ckpt_fold"/*; do
#         ckpt=$(basename "$file")
#         ckpt_num=${ckpt:0:5}

#         # ckpt_num이 [00001, 00010] 범위인지 확인
#         if [[ $ckpt_num =~ ^(0000[1-9]|00010)$ ]]; then
#             case "$model" in
#                 "HiCARN")
#                     python model_HiCARN/40x40_Predict.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}
#                     ;;
#                 "DeepHiC")
#                     python model_DeepHiC/data_predict.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}
#                     ;;
#                 "HiCNN")
#                     python model_HiCNN2/HiCNN2_predict.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}
#                     ;;
#                 "DFHiC")
#                     python model_DFHiC/run_predict.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}
#                     ;;
#                 "SRHiC")
#                     python model_SRHiC/src/SRHiC_predict.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}
#                     ;;
#                 "HiCPlus")
#                     python model_HiCPlus/HiCPlus/pred_chromosome.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}
#                     ;;
#                 "iEnhance")
#                     python model_iEnhance/predict-hic.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}
#                     ;;
#                 *)
#                     echo "Unknown model: ${model}"
#                     ;;
#             esac
#         fi
#     done
#     echo "Finished predictions for model: ${model}"
# done
