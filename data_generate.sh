#!/bin/bash
seed=42
root_dir=$(pwd)
# bash data_generate.sh -i ./data_KR -d ./data_KR_downsampled_16 -m iEnhance -g ./hg19.txt -r 16 -o ./ -s 300 -n KR

while getopts ":i:d:m:g:r:o:s:n:t:v:p:" flag; do
    case $flag in
        i) input_data_dir=$(echo "${OPTARG}" | sed 's#/*$##');;
        d) input_downsample_dir=$(echo "${OPTARG}" | sed 's#/*$##');;
        m) model=$OPTARG;;
        g) ref_chrom=$OPTARG;;
        r) down_ratio=$OPTARG;;
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
if [ -z "${input_data_dir}" ] || [ -z "${input_downsample_dir}" ] || [ -z "${model}" ] || [ -z "${ref_chrom}" ] || [ -z "${down_ratio}" ] || [ -z "${output_dir}" ]; then
    echo "Usage: $0 -i <input_data_path> -d <input_downsample_path> -m <model_name> -g <ref_chromosome_length> -r <downsample_ratio> -o <output_path> -s <max_value> -n normalization -t <train_set_chromosome> -v <valid_set_chromosome> -p <prediction_set_chromosome>" >&2
    exit 1
fi

# 데이터셋 split 기본값 설정
if [ "${model}" = "hicplus" ]; then
    train_set=${train_set:-"1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17"}
    prediction_set=${prediction_set:-"18 19 20 21 22"}
    echo ""
    echo "  ...For training set, chromosome ${train_set}"
    echo "     For test set, chromosome ${prediction_set}"
elif [[ "${model}" =~ ^(HiCNN2|SRHiC|deepHiC|HiCARN|DFHiC|iEnhance)$ ]]; then
    train_set=${train_set:-"1 2 3 4 5 6 7 8 9 10 11 12 13 14"}
    valid_set=${valid_set:-"15 16 17"}
    prediction_set=${prediction_set:-"18 19 20 21 22"}
    echo ""
    echo "  ...For training set, chromosome ${train_set}" 
    echo "     For validation set, chromosome ${valid_set}"
    echo "     For test set, chromosome ${prediction_set}"
else
    echo "Model name should be one of the hicplus, HiCNN2, SRHiC, deepHiC, HiCARN, DFHiC, and iEnhance."
    exit 1
fi

# 저장 디렉토리 존재하지 않는 경우 생성
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

echo ""
echo "  ...Start generating input data for ${model} training..."
echo "     using Hi-C data of ${input_data_dir} and 1/${down_ratio} downsampled data of ${input_downsample_dir}"
echo ""

# Python 스크립트 비버퍼링 모드로 실행
if [ "${model}" = "iEnhance" ]; then
    python -u model_iEnhance/divide-data.py -i "${input_data_dir}" -d "${input_downsample_dir}" -m "${model}" -g "${ref_chrom}" -r "${down_ratio}" -o "${output_dir}" -n "${normalization}" -s "${max_value}" -t "${train_set}" -v "${valid_set}" -p "${prediction_set}"
    # python -u model_iEnhance/construct_sets.py -i "${output_dir}/data_${model}/chrs_${normalization}_${max_value}/" -m "${model}" -r "${down_ratio}" -o "${output_dir}" -n "${normalization}" -s "${max_value}" -t "${train_set}" -v "${valid_set}" -p "${prediction_set}"
else
    python -u data_generate.py -i "${input_data_dir}" -d "${input_downsample_dir}" -m "${model}" -g "${ref_chrom}" -r "${down_ratio}" -o "${output_dir}/" -n "${normalization}" -s "${max_value}" -t "${train_set}" -v "${valid_set}" -p "${prediction_set}" -s "${max_value}" -n "${KR}"
fi
