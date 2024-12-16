#!/bin/bash
seed=13
root_dir=$(pwd)

# bash data_generate_for_prediction.sh -i "./data/MAT/GM12878__10.2M_10Kb_KR/" -b "10000" -m "DFHiC" -g "./hg19.txt" -o "./data_model" -s "300"

while getopts ":i:b:m:g:o:s:" flag; do
    case $flag in
        i) input_data_dir=$(echo "${OPTARG}" | sed 's#/*$##');;
        b) bin_size=$OPTARG;;
        m) model=$OPTARG;;
        g) ref_genome=$OPTARG;;
        o) saved_in=$(echo "${OPTARG}" | sed 's#/*$##');;
        s) max_value=$OPTARG;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1;;
    esac
done

# 필수 인자 체크
if [ -z "${input_data_dir}" ] || [ -z "${bin_size}" ] || [ -z "${model}" ] || [ -z "${ref_genome}" ] || [ -z "${saved_in}" ] || [ -z "${max_value}" ]; then
    echo "Usage: $0 -i <input_data_path> -b <resolution> -m <model_name> -g <ref_genome> -o <saved_in> -s <max_value> " >&2
    exit 1
fi

if [[ "${model}" =~ ^(HiCPlus|HiCNN|SRHiC|DeepHiC|HiCARN|DFHiC|iEnhance)$ ]]; then
    if [ ! -d "${saved_in}/data_${model}" ]; then
        mkdir -p "${saved_in}/data_${model}"
    fi
    echo ""
    echo "  ...Start generating input data for ${model} prediction..."
    echo "     using Hi-C data of ${input_data_dir}/"
    echo ""
else
    echo "Model name should be one of the HiCPlus, HiCNN, SRHiC, DeepHiC, HiCARN, DFHiC, and iEnhance."
    exit 1
fi

# iEnhance 모델 전용 작업
if [ "${model}" = "iEnhance" ]; then
    python3 -u model_iEnhance/divide-data.py -a "Enhancement" -i "${input_data_dir}" -m "${model}" -b "${bin_size}" -g "${ref_genome}" -o "${saved_in}/data_${model}/" -s "${max_value}"
else
    python3 -u data_generate_for_prediction.py -i "${input_data_dir}" -b "${bin_size}" -m "${model}" -g "${ref_genome}" -o "${saved_in}/data_${model}/" -s "${max_value}"
fi
