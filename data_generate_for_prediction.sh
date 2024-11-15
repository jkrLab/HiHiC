#!/bin/bash
seed=42
root_dir=$(pwd)

# bash data_generate_for_prediction.sh -i /project/HiHiC/data_GM12878 -m iEnhance -g ./hg19.txt -o ./ -s 300 -n KR

while getopts ":e:i:m:g:o:n:s:" flag; do
    case $flag in
        e) explain=$OPTARG;;
        i) input_data_dir=$(echo "${OPTARG}" | sed 's#/*$##');;
        m) model=$OPTARG;;
        g) ref_chrom=$OPTARG;;
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

# normalization 기본값 설정
if [ -z "${normalization}" ]; then
    normalization="Unknown"
fi

# normalization 기본값 설정
if [ -z "${explain}" ]; then
    explain=""
fi

# 필수 인자 체크
if [ -z "${input_data_dir}" ] || [ -z "${model}" ] || [ -z "${ref_chrom}" ] || [ -z "${output_dir}" ] || [ -z "${max_value}" ]; then
    echo "Usage: $0 -i <input_data_path> -m <model_name> -g <ref_chromosome_length> -o <output_path> -s <max_value> [-n <normalization>]" >&2
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

# iEnhance 모델 전용 작업
if [ "${model}" = "iEnhance" ]; then
    prediction_set=()
    for file in $(ls "${input_data_dir}"); do
        prefix=$(basename "$file" | awk -F'_' '{print $1}')  # '_' 기준으로 앞부분 추출
        prediction_set+=("$prefix")
    done

    # 배열을 쉼표로 구분된 문자열로 변환
    prediction_set_str=$(IFS=' '; echo "${prediction_set[*]}")

    python -u model_iEnhance/divide-data.py -a "Enhancement" -i "${input_data_dir}" -m "${model}" -g "${ref_chrom}" -o "${output_dir}" -n "${normalization}" -s "${max_value}" -p "${prediction_set_str}" -e "${explain}"
    python -u model_iEnhance/construct_sets.py -a "Enhancement" -i "${output_dir}/data_${model}/chrs_${normalization}_${max_value}/" -m "${model}" -o "${output_dir}" -n "${normalization}" -s "${max_value}" -p "${prediction_set_str}" -e "${explain}"
else
    python -u data_generate_for_prediction.py -i "${input_data_dir}" -m "${model}" -g "${ref_chrom}" -o "${output_dir}/" -n "${normalization}" -s "${max_value}" -e "${explain}"
fi
