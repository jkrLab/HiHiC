#!/bin/bash
seed=42
root_dir=$(pwd)

# example)
# bash data_generate.sh -i ./data_KR -d ./data_KR_downsampled_16 -b 10000 -m iEnhance -g ./hg19.txt -r 5000000 -o ./ -s 300 -n KR -t "1 2 3 4 5 6 7 8 9 10 11 12 13 14" -v "15 16 17" -p "18 19 20 21 22"

while getopts ":i:d:b:m:g:r:o:s:n:t:v:p:" flag; do
    case $flag in
        i) input_data_dir=$(echo "${OPTARG}" | sed 's#/*$##');;
        d) input_downsample_dir=$(echo "${OPTARG}" | sed 's#/*$##');;
        b) bin_size=$OPTARG;;
        m) model=$OPTARG;;
        g) ref_chrom=$OPTARG;;
        r) read=$OPTARG;;
        o) output_dir=$(echo "${OPTARG}" | sed 's#/*$##');;
        s) max_value=$OPTARG;;
        n) normalization=$OPTARG;;
        t) train_set=$OPTARG;;
        v) valid_set=$OPTARG;;
        p) prediction_set=$OPTARG;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1;;
    esac
done

# 필수 인자 체크
if [ -z "${input_data_dir}" ] || [ -z "${input_downsample_dir}" ] || [ -z "${bin_size}" ] || [ -z "${model}" ] || [ -z "${ref_chrom}" ] || [ -z "${read}" ] || [ -z "${output_dir}" ]|| [ -z "${train_set}" ] || [ -z "${prediction_set}" ]; then
    echo "Usage: $0 -i <input_data_path> -d <input_downsample_path> -b <bin_size> -m <model_name> -g <ref_chromosome_length> -r <downsampled_read> -o <output_path> -s <max_value> -n normalization -t <train_set_chromosome> -v <valid_set_chromosome> -p <prediction_set_chromosome>" >&2
    exit 1
fi

if [[ "${model}" != "HiCPlus" ]] && [ -z "${valid_set}" ]; then
    echo "For model ${model}, -v <valid_set> is mandatory." >&2
    exit 1
fi

# 데이터셋 split 기본값 설정
if [ "${model}" = "HiCPlus" ]; then
    echo ""
    echo "  ...For training set, chromosome ${train_set} ${valid_set}"
    echo "     For test set, chromosome ${prediction_set}"
elif [[ "${model}" =~ ^(HiCNN|SRHiC|DeepHiC|HiCARN|DFHiC|iEnhance)$ ]]; then
    echo ""
    echo "  ...For training set, chromosome ${train_set}" 
    echo "     For validation set, chromosome ${valid_set}"
    echo "     For test set, chromosome ${prediction_set}"
else
    echo "Model name should be one of the HiCPlus, HiCNN, SRHiC, DeepHiC, HiCARN, DFHiC, and iEnhance."
    exit 1
fi

# 저장 디렉토리 존재하지 않는 경우 생성
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

echo ""
echo "  ...Start generating ${bin_size} bp resolution input data for ${model} training..."
echo "     using Hi-C data of ${input_data_dir}/ and ${read} downsampled read data of ${input_downsample_dir}/"
echo ""

# Python 스크립트 비버퍼링 모드로 실행
if [ "${model}" = "iEnhance" ]; then
    python -u model_iEnhance/divide-data.py -a "Train" -i "${input_data_dir}" -d "${input_downsample_dir}" -b "${bin_size}" -m "${model}" -g "${ref_chrom}" -r "${read}" -o "${output_dir}" -n "${normalization}" -s "${max_value}" -t "${train_set}" -v "${valid_set}" -p "${prediction_set}"
    python -u model_iEnhance/construct_sets.py -a "Train" -i "${output_dir}/data_${model}/chrs_${read}_${bin_size}/" -b "${bin_size}" -m "${model}" -r "${read}" -o "${output_dir}" -n "${normalization}" -s "${max_value}" -t "${train_set}" -v "${valid_set}" -p "${prediction_set}"
else
    python -u data_generate_for_training.py -i "${input_data_dir}" -d "${input_downsample_dir}" -b "${bin_size}" -m "${model}" -g "${ref_chrom}" -r "${read}" -o "${output_dir}/" -n "${normalization}" -s "${max_value}" -t "${train_set}" -v "${valid_set}" -p "${prediction_set}"
fi