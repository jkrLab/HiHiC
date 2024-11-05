#!/bin/bash

seed=42
root_dir=$(pwd)
gpu_id=-1

while getopts ":m:c:b:g:r:i:o:" flag; 
do
    case $flag in
        m) model=$OPTARG;;
        c) ckpt_file=$OPTARG;;
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

# 파일 존재 여부를 확인하는 함수
check_file_exists() {
    if [ ! -f "$1" ]; then
        echo "Error: The file '$1' does not exist."
        exit 1
    fi
}

# 필수 인수인 --model 및 --ckpt_file이 제공되었는지 확인합니다.
if [ -z "$model" ] || [ -z "$ckpt_file" ]; then
    echo "Error: Both --model and --ckpt_file are required."
    exit 1
fi

check_file_exists "$ckpt_file"
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
echo "      The pretrained model, ${ckpt_file} will be utilized."
echo "      If available, GPU:${gpu_id} will be used for prediction; otherwise, CPU will be used."
echo ""

if [ ${model} = "HiCARN1" ]; then 
    python model_HiCARN/40x40_Predict.py --root_dir ${root_dir} --model ${model} --ckpt_file ${ckpt_file} --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}  

elif [ ${model} = "HiCARN2" ]; then 
    python model_HiCARN/40x40_Predict.py --root_dir ${root_dir} --model ${model} --ckpt_file ${ckpt_file} --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}

elif [ ${model} = "DeepHiC" ]; then 
    python model_DeepHiC/data_predict.py --root_dir ${root_dir} --model ${model} --ckpt_file ${ckpt_file} --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}

elif [ ${model} = "HiCNN2" ]; then 
    python model_HiCNN2/HiCNN2_predict.py --root_dir ${root_dir} --model ${model} --ckpt_file ${ckpt_file} --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}

elif [ ${model} = "DFHiC" ]; then 
    python model_DFHiC/run_predict.py --root_dir ${root_dir} --model ${model} --ckpt_file ${ckpt_file} --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}

elif [ ${model} = "SRHiC" ]; then 
    python model_SRHiC/src/SRHiC_predict.py --root_dir ${root_dir} --model ${model} --ckpt_file ${ckpt_file} --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}

elif [ ${model} = "HiCPlus" ]; then
    python model_HiCPlus/HiCPlus/pred_chromosome.py --root_dir ${root_dir} --model ${model} --ckpt_file ${ckpt_file} --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}

elif [ ${model} = "iEnhance" ]; then
    python model_iEnhance/predict-hic.py --root_dir ${root_dir} --model ${model} --ckpt_file ${ckpt_file} --batch_size ${batch_size} --gpu_id ${gpu_id} --down_ratio ${down_ratio} --input_data ${input_data} --output_data_dir ${output_data_dir}

else
    echo "Model name should be one of the DeepHiC, HiCNN2, DFHiC, HiCPlus, HiCARN1, HiCARN2, SRHiC, iEnhance."

fi