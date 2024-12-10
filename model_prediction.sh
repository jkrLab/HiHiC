#!/bin/bash

seed=13
root_dir=$(pwd)
gpu_id=0		
batch_size=16


# bash model_prediction.sh -m DFHiC -c best_ckpt.npz -b 16 -g 0 -i /HiHiC/data_DFHiC/enhance/GM12878_10.2M_10Kb_KR.npz -o ./data_model_out

while getopts ":m:c:b:g:i:o:e:" flag; 
do
    case $flag in
        m) model=$OPTARG;;
        c) ckpt_file=$OPTARG;;
        b) batch_size=$OPTARG;;
        g) gpu_id=$OPTARG;;
        i) input_data=$OPTARG;;
        o) saved_in=$OPTARG;;
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
if [ -z "${model}" ] || [ -z "${ckpt_file}" ] || [ -z "${input_data}" ] || [ -z "${saved_in}" ]; then
    echo "Usage: $0 -m <model_name> -c <ckechpoint> -i <input_data>  -o <saved_in> " >&2
    exit 1
fi

check_file_exists "${ckpt_file}"
check_file_exists "${input_data}"


# 저장 디렉토리 존재하지 않는 경우 생성
if [ ! -d "${saved_in}/OUTPUT" ]; then
    mkdir -p "${saved_in}/OUTPUT"
fi

echo ""
echo "  ...Current working directory is ${root_dir}."
echo "      ${model} will enhance low resolution HiC data, ${input_data}"
echo "      And the enhanced HiC data will be saved in ${saved_in}/OUTPUT." 
echo "      The pretrained model, ${ckpt_file} will be utilized."
echo "      If available, GPU:${gpu_id} will be used for prediction; otherwise, CPU will be used."
echo ""

if [ ${model} = "HiCARN1" ]; then 
    python model_HiCARN/40x40_Predict.py --root_dir "${root_dir}" --model "${model}" --ckpt_file "${ckpt_file}" --batch_size "${batch_size}" --gpu_id "${gpu_id}" --input_data "${input_data}" --output_data_dir "${saved_in}/OUTPUT"

elif [ ${model} = "HiCARN2" ]; then 
    python model_HiCARN/40x40_Predict.py --root_dir "${root_dir}" --model "${model}" --ckpt_file "${ckpt_file}" --batch_size "${batch_size}" --gpu_id "${gpu_id}" --input_data "${input_data}" --output_data_dir "${saved_in}/OUTPUT"

elif [ ${model} = "DeepHiC" ]; then 
    python model_DeepHiC/data_predict.py --root_dir "${root_dir}" --model "${model}" --ckpt_file "${ckpt_file}" --batch_size "${batch_size}" --gpu_id "${gpu_id}" --input_data "${input_data}" --output_data_dir "${saved_in}/OUTPUT"

elif [ ${model} = "HiCNN2" ]; then 
    python model_HiCNN2/HiCNN2_predict.py --root_dir "${root_dir}" --model "${model}" --ckpt_file "${ckpt_file}" --batch_size "${batch_size}" --gpu_id "${gpu_id}" --input_data "${input_data}" --output_data_dir "${saved_in}/OUTPUT"

elif [ ${model} = "DFHiC" ]; then 
    python model_DFHiC/run_predict.py --root_dir "${root_dir}" --model "${model}" --ckpt_file "${ckpt_file}" --batch_size "${batch_size}" --gpu_id "${gpu_id}" --input_data "${input_data}" --output_data_dir "${saved_in}/OUTPUT"

elif [ ${model} = "SRHiC" ]; then 
    python model_SRHiC/src/SRHiC_predict.py --root_dir "${root_dir}" --model "${model}" --ckpt_file "${ckpt_file}" --batch_size "${batch_size}" --gpu_id "${gpu_id}" --input_data "${input_data}" --output_data_dir "${saved_in}/OUTPUT"

elif [ ${model} = "HiCPlus" ]; then
    python model_HiCPlus/hicplus/pred_chromosome.py --root_dir "${root_dir}" --model "${model}" --ckpt_file "${ckpt_file}" --batch_size "${batch_size}" --gpu_id "${gpu_id}" --input_data "${input_data}" --output_data_dir "${saved_in}/OUTPUT"

elif [ ${model} = "iEnhance" ]; then
    prefix=${input_data%%__*}
    python model_iEnhance/predict-hic.py --root_dir "${root_dir}" --model "${model}" --ckpt_file "${ckpt_file}" --batch_size "${batch_size}" --gpu_id "${gpu_id}" --input_data "${input_data}" --output_data_dir "${saved_in}/${prefix}"

else
    echo "Model name should be one of the DeepHiC, HiCNN2, DFHiC, HiCPlus, HiCARN1, HiCARN2, SRHiC, iEnhance."

fi
