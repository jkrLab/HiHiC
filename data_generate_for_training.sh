#!/bin/bash
seed=13
root_dir=$(pwd)

# example)
# bash data_generate.sh -i ./data/GM12848/MAT/GM12878_5M_10Kb_KR -d ./data/GM12848/MAT/GM12878_2M_10Kb_KR -b 10000 -m iEnhance -g ./hg19.txt -o ./data_model -s 300 -t "1 2 3 4 5 6 7 8 9 10 11 12 13 14" -v "15 16 17" -p "18 19 20 21 22"

while getopts ":i:d:r:b:m:g:o:s:t:v:p:" flag; do
    case $flag in
        i) input_data_dir=$(echo "${OPTARG}" | sed 's#/*$##');;
        d) input_downsample_dir=$(echo "${OPTARG}" | sed 's#/*$##');;
        r) reads=$OPTARG;;
        b) bin_size=$OPTARG;;
        m) model=$OPTARG;;
        g) ref_genome=$OPTARG;;
        o) saved_in=$(echo "${OPTARG}" | sed 's#/*$##');;
        s) max_value=$OPTARG;;
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
if [ -z "${input_data_dir}" ] || [ -z "${input_downsample_dir}" ] || [ -z "${bin_size}" ] || [ -z "${model}" ] || [ -z "${ref_genome}" ] || [ -z "${saved_in}" ]|| [ -z "${train_set}" ] || [ -z "${prediction_set}" ]; then
    echo "Usage: $0 -i <input_data_path> -d <input_downsample_path> -b <resolution> -m <model_name> -g <ref_genome> -o <saved_in> -s <max_value> -t <train_set_chromosome> -v <valid_set_chromosome> -p <prediction_set_chromosome>" >&2
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
if [ ! -d "${saved_in}/data_${model}" ]; then
    mkdir -p "${saved_in}/data_${model}"
fi

echo ""
echo "  ...Start generating input data for ${model} training..."
echo "     using Hi-C data of ${input_data_dir}/ and downsampled data of ${input_downsample_dir}/"
echo ""

# Python 스크립트 비버퍼링 모드로 실행
if [ "${model}" = "iEnhance" ]; then
    python3 -u model_iEnhance/divide-data.py -a "Train" -i "${input_data_dir}" -d "${input_downsample_dir}" -r "${reads}" -b "${bin_size}" -m "${model}" -g "${ref_genome}" -o "${saved_in}/data_${model}/" -s "${max_value}" -t "${train_set}" -v "${valid_set}" -p "${prediction_set}"
    python3 -u model_iEnhance/construct_sets.py -a "Train" -i "${saved_in}/data_${model}/chrs" -b "${bin_size}" -m "${model}" -o "${saved_in}/data_${model}/" -s "${max_value}" -t "${train_set}" -v "${valid_set}" -p "${prediction_set}"
else
    python3 -u data_generate_for_training.py -i "${input_data_dir}" -d "${input_downsample_dir}" -b "${bin_size}" -m "${model}" -g "${ref_genome}" -o "${saved_in}/data_${model}/" -s "${max_value}" -t "${train_set}" -v "${valid_set}" -p "${prediction_set}"
fi
