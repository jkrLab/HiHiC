#!/bin/bash

seed=42
root_dir=$(pwd)

while getopts ":m:c:b:g:r:i:o:" flag; 
do
    case $flag in
        m) model=$OPTARG;;
        c) ckpt_fold=$OPTARG;;
        b) batch_size=$OPTARG;;
        g) gpu_id=$OPTARG;;
        r) down_ratio=$OPTARG;; 
        i) input_data=$OPTARG;;
        o) output_data_dir=$OPTARG;;
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
            remainder=$((10#$ckpt_num % 100))

            if [ "$remainder" -eq 0 ]; then
                python model_HiCARN/40x40_Predict.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}
            fi
        fi
    done

elif [ "$model" = "HiCARN2" ]; then
    for file in "$ckpt_fold"/*; do
        ckpt=$(basename "$file")
        ckpt_num=${ckpt:0:5}

        if [[ $ckpt_num =~ ^[0-9]+$ ]]; then
            remainder=$((10#$ckpt_num % 100))

            if [ "$remainder" -eq 0 ]; then
                python model_HiCARN/40x40_Predict.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}
            fi
        fi
    done

elif [ "$model" = "DeepHiC" ]; then
    for file in "$ckpt_fold"/*; do
        ckpt=$(basename "$file")
        ckpt_num=${ckpt:0:5}

        if [[ $ckpt_num =~ ^[0-9]+$ ]]; then
            remainder=$((10#$ckpt_num % 100))

            if [ "$remainder" -eq 0 ]; then
                python model_DeepHiC/data_predict.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}
            fi        
        fi
    done

elif [ "$model" = "HiCNN2" ]; then
    for file in "$ckpt_fold"/*; do
        ckpt=$(basename "$file")
        ckpt_num=${ckpt:0:5}

        if [[ $ckpt_num =~ ^[0-9]+$ ]]; then
            remainder=$((10#$ckpt_num % 100))

            if [ "$remainder" -eq 0 ]; then
                python model_HiCNN2/HiCNN2_predict.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}
            fi
        fi
    done

elif [ "$model" = "DFHiC" ]; then
    for file in "$ckpt_fold"/*; do
        ckpt=$(basename "$file")
        ckpt_num=${ckpt:0:5}

        if [[ $ckpt_num =~ ^[0-9]+$ ]]; then
            remainder=$((10#$ckpt_num % 100))

            if [ "$remainder" -eq 0 ]; then
                python model_DFHiC/run_predict.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}
            fi
        fi
    done

elif [ "$model" = "SRHiC" ]; then
    for file in $(find "$ckpt_fold" -type f -name '*.meta'); do
        ckpt=$(basename "$file")
        ckpt_num=${ckpt:0:5}

        if [[ $ckpt_num =~ ^[0-9]+$ ]]; then
            remainder=$((10#$ckpt_num % 100))

            if [ "$remainder" -eq 0 ]; then
                python model_SRHiC/src/SRHiC_predict.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}
            fi
        fi
    done

elif [ "$model" = "hicplus" ]; then
    for file in "$ckpt_fold"/*; do
        ckpt=$(basename "$file")
        ckpt_num=${ckpt:0:5}

        if [[ $ckpt_num =~ ^[0-9]+$ ]]; then
            remainder=$((10#$ckpt_num % 100))

            if [ "$remainder" -eq 0 ]; then
                python model_hicplus/hicplus/pred_chromosome.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}
            fi
        fi
    done

elif [ "$model" = "iEnhance" ]; then
    for file in "$ckpt_fold"/*; do
        ckpt=$(basename "$file")
        ckpt_num=${ckpt:0:5}

        if [[ $ckpt_num =~ ^[0-9]+$ ]]; then
            remainder=$((10#$ckpt_num % 100))

            if [ "$remainder" -eq 0 ]; then
                python model_iEnhance/predict-hic.py --root_dir ${root_dir} --model ${model} --ckpt_file "${ckpt_fold}/${ckpt}" --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}
            fi
        fi
    done

else
    echo "Model name should be one of the DeepHiC, HiCNN2, DFHiC, hicplus, HiCARN1, HiCARN2, SRHiC, iEnhance."
fi
