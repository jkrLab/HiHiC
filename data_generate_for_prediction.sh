#!/bin/bash
seed=42
root_dir=$(pwd)
# bash data_generate_for_prediction.sh -i ./data_KR -m iEnhance -g ./hg19.txt -r 16 -o ./ -s 300 -n KR -t "1 2 3 4 5 6 7 8 9 10 11 12 13 14" -v "15 16 17" -p "18 19 20 21 22"

while getopts ":i:m:g:r:o:n:s:" flag; do
    case $flag in
        i) input_data_dir=$(echo "${OPTARG}" | sed 's#/*$##');;
        m) model=$OPTARG;;
        g) ref_chrom=$OPTARG;;
        r) down_ratio=$OPTARG;;
        o) output_dir=$(echo "${OPTARG}" | sed 's#/*$##');;
        n) normalization=$OPTARG;;
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
if [ -z "${input_data_dir}" ] || [ -z "${model}" ] || [ -z "${ref_chrom}" ] || [ -z "${output_dir}" ]; then
    echo "Usage: $0 -i <input_data_path> -m <model_name> -g <ref_chromosome_length> -o <output_path> -s <max_value> -n <normalization>" >&2
    exit 1
fi

if [[ "${model}" =~ ^(HiCPlus|HiCNN|SRHiC|DeepHiC|HiCARN|DFHiC|iEnhance)$ ]]; then
    if [ ! -d "${output_dir}" ]; then
        mkdir -p "${output_dir}"
    fi
    echo ""
    echo "  ...Start generating input data for ${model} prediction..."
    echo "     using Hi-C data of ${input_data_dir}/"
    echo ""
else
    echo "Model name should be one of the HiCPlus, HiCNN, SRHiC, DeepHiC, HiCARN, DFHiC, and iEnhance."
    exit 1
fi


# Python 스크립트 비버퍼링 모드로 실행
if [ "${model}" = "iEnhance" ]; then
    python -u model_iEnhance/divide-data.py -i "${input_data_dir}" -d "${input_downsample_dir}" -m "${model}" -g "${ref_chrom}" -r "${down_ratio}" -o "${output_dir}" -n "${normalization}" -s "${max_value}" -t "${train_set}" -v "${valid_set}" -p "${prediction_set}"
    python -u model_iEnhance/construct_sets.py -i "${output_dir}/data_${model}/chrs_${normalization}_${max_value}/" -m "${model}" -r "${down_ratio}" -o "${output_dir}" -n "${normalization}" -s "${max_value}" -t "${train_set}" -v "${valid_set}" -p "${prediction_set}"
else
    python -u data_generate_for_prediction.py -i "${input_data_dir}" -m "${model}" -g "${ref_chrom}"  -o "${output_dir}/" -n "${normalization}" -s "${max_value}" 
fi
