# #!/bin/bash
set -euo pipefail

# root_dir=$(pwd)
# output_dir="${root_dir}/whole_mats_output"  # 최종 결과 저장 디렉토리 설정
# models=("HiCARN" "DeepHiC" "HiCNN" "HiCPlus" "iEnhance" "DFHiC" "SRHiC")

# mkdir -p "$output_dir"  # 결과 저장 폴더 생성

# for model in "${models[@]}"; do
#     input_data_dir="${root_dir}/output_${model}"  # 각 모델의 예측 출력 디렉토리 경로
    
#     if [ -d "$input_data_dir" ]; then
#         echo "Processing output for model: ${model}"
#         python data_make_whole.py -i "$input_data_dir" -m "${model}" -o "$output_dir"
#     else
#         echo "Warning: Directory $input_data_dir does not exist. Skipping model ${model}."
#     fi
# done

# echo "All model outputs processed. Results are saved in $output_dir."

#!/bin/bash

# 필수 경로 및 설정
output_dir="/project/HiHiC/output_whole_mat"
input_data_base="/project/HiHiC/output_"

# 모델 이름 목록
models=("HiCARN" "DeepHiC" "HiCNN" "HiCPlus" "DFHiC" "SRHiC")

# 반복 수행
for model in "${models[@]}"; do
    input_data_dir="${input_data_base}${model}"
    input_files=()
    
    # 파일 범위 내에서 검색하여 배열에 추가
    for i in $(seq -w 1 10); do
        input_file="${input_data_dir}/${model}_predict_16_000${i}.npz"
        if [ -f "$input_file" ]; then
            input_files+=("$input_file")
        fi
    done

    # data_make_whole.py 실행
    if [ ${#input_files[@]} -gt 0 ]; then
        python data_make_whole.py -i "${input_files[@]}" -m "$model" -o "$output_dir"
    else
        echo "No files found for model: $model in the specified range."
    fi
done
