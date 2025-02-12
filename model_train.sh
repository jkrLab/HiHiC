#!/bin/bash
set -euo pipefail

seed=13
root_dir=$(pwd)


# 임시 파일 설정
gpu_mem_file="/tmp/gpu_mem_usage.tmp"
ram_mem_file="/tmp/ram_mem_usage.tmp"

# 초기화
echo 0 > ${gpu_mem_file}
echo 0 > ${ram_mem_file}

# GPU 및 RAM 메모리 사용량을 기록하는 함수
monitor_resources() {
    while true; do
        # GPU가 설정된 경우 GPU 메모리 사용량 추출
        if [[ -n "${gpu_id}" ]]; then
            gpu_memory=$(nvidia-smi --id=${gpu_id} --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
            if [[ $? -ne 0 ]]; then
                echo "  ...Failed to use GPU; CPU is utilized." >&2
                gpu_memory=0
            else
                gpu_memory=$(echo ${gpu_memory} | tr -d ' ')
            fi
        else
            gpu_memory=0
        fi
        
        # RAM 메모리 사용량 추출
        ram_memory=$(free -m | awk '/Mem:/ {print $3}' | tr -d ' ')

        # 최대 GPU 메모리 사용량 계산
        max_gpu_memory=$(cat ${gpu_mem_file})
        if (( gpu_memory > max_gpu_memory )); then
            echo $gpu_memory > ${gpu_mem_file}
        fi
        
        # 최대 RAM 사용량 계산
        max_ram_memory=$(cat ${ram_mem_file})
        if (( ram_memory > max_ram_memory )); then
            echo $ram_memory > ${ram_mem_file}
        fi

        sleep 5  # 5초마다 체크
    done
}

# 학습 스크립트 옵션 처리
while getopts ":m:e:b:g:o:l:t:v:" flag; 
do
    case $flag in
        m) model=$OPTARG;;
        e) epoch=$OPTARG;;
        b) batch_size=$OPTARG;;
        g) gpu_id=$OPTARG;;
        o) output_model_dir=$OPTARG;;
        l) loss_log_dir=$OPTARG;;
        t) train_data_dir=$OPTARG;;
        v) valid_data_dir=$OPTARG;;
        \?)
        echo "Invalid option: -$OPTARG" >&2
        exit 1;;
        :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1;;
    esac
done

echo ""
echo "  ...Current working directory is ${root_dir}."
echo "      ${model} will train for ${epoch} epochs"
echo "      Training data in ${train_data_dir}, ${valid_data_dir} will be used." 
echo "      After training, the models will be saved in ${output_model_dir}/checkpoints_${model}."
echo "      Loss variations and memory usage will be saved in ${loss_log_dir}."
echo "      If a GPU is available, it will be utilized - GPU:${gpu_id}."
echo ""

# validation set 유효성 검사 (HiCPlus 제외)
if [ -z "${valid_data_dir}" ] && [ "${model}" != "HiCPlus" ]; then
    echo "Error: Validation data directory is required for ${model} model." >&2
    exit 1
fi

mkdir -p ${loss_log_dir}

# 백그라운드에서 GPU 및 RAM 모니터링 시작
monitor_resources &  # 백그라운드에서 실행
monitor_pid=$!  # 백그라운드 프로세스의 PID 저장

# 학습 시작 시간 기록
start_time=$(date +%s)
start_datetime=$(date "+%Y-%m-%d %H:%M:%S")

# 모델에 따라 학습 실행
case ${model} in
    HiCPlus)
        python model_HiCPlus/hicplus/train_models.py --root_dir "${root_dir}" --model "${model}" --epoch "${epoch}" --batch_size "${batch_size}" --gpu_id "${gpu_id}" --output_model_dir "${output_model_dir}/checkpoints_${model}" --loss_log_dir "${loss_log_dir}" --train_data_dir "${train_data_dir}"
        ;;
    HiCNN)
        python model_HiCNN2/HiCNN2_training.py --root_dir "${root_dir}" --model "${model}" --epoch "${epoch}" --batch_size "${batch_size}" --gpu_id "${gpu_id}" --output_model_dir "${output_model_dir}/checkpoints_${model}" --loss_log_dir "${loss_log_dir}" --train_data_dir "${train_data_dir}" --valid_data_dir "${valid_data_dir}"
        ;;
    SRHiC)
        python model_SRHiC/src/SRHiC_main.py --root_dir "${root_dir}" --model "${model}" --epoch "${epoch}" --batch_size "${batch_size}" --gpu_id "${gpu_id}" --output_model_dir "${output_model_dir}/checkpoints_${model}" --loss_log_dir "${loss_log_dir}" --train_data_dir "${train_data_dir}" --valid_data_dir "${valid_data_dir}" --train True
        ;;
    DeepHiC)
        python model_DeepHiC/train.py --root_dir "${root_dir}" --model "${model}" --epoch "${epoch}" --batch_size "${batch_size}" --gpu_id "${gpu_id}" --output_model_dir "${output_model_dir}/checkpoints_${model}" --loss_log_dir "${loss_log_dir}" --train_data_dir "${train_data_dir}" --valid_data_dir "${valid_data_dir}"
        ;;
    HiCARN1)
        python model_HiCARN/HiCARN_1_Train.py --root_dir "${root_dir}" --model "${model}" --epoch "${epoch}" --batch_size "${batch_size}" --gpu_id "${gpu_id}" --output_model_dir "${output_model_dir}/checkpoints_${model}" --loss_log_dir "${loss_log_dir}" --train_data_dir "${train_data_dir}" --valid_data_dir "${valid_data_dir}"    
        ;;
    HiCARN2)
        python model_HiCARN/HiCARN_2_Train.py --root_dir "${root_dir}" --model "${model}" --epoch "${epoch}" --batch_size "${batch_size}" --gpu_id "${gpu_id}" --output_model_dir "${output_model_dir}/checkpoints_${model}" --loss_log_dir "${loss_log_dir}" --train_data_dir "${train_data_dir}" --valid_data_dir "${valid_data_dir}"
        ;;
    DFHiC)
        python model_DFHiC/run_train.py --root_dir "${root_dir}" --model "${model}" --epoch "${epoch}" --batch_size "${batch_size}" --gpu_id "${gpu_id}" --output_model_dir "${output_model_dir}/checkpoints_${model}" --loss_log_dir "${loss_log_dir}" --train_data_dir "${train_data_dir}" --valid_data_dir "${valid_data_dir}"
        ;;
    iEnhance)
        python model_iEnhance/train.py --train_option 0 --root_dir "${root_dir}" --model "${model}" --epoch "${epoch}" --batch_size "${batch_size}" --gpu_id "${gpu_id}" --output_model_dir "${output_model_dir}/checkpoints_${model}" --loss_log_dir "${loss_log_dir}" --train_data_dir "${train_data_dir}" --valid_data_dir "${valid_data_dir}"
        ;;
    *)
        echo "Model name should be one of: HiCPlus, HiCNN, SRHiC, DeepHiC, HiCARN1, HiCARN2, DFHiC, iEnhance." >&2
        exit 1
        ;;
esac

# 학습 종료 시간 기록
end_time=$(date +%s)
end_datetime=$(date "+%Y-%m-%d %H:%M:%S")

# 총 학습 시간 계산
total_time=$((${end_time} - ${start_time}))

days=$((${total_time} / 86400))
hours=$(((${total_time} % 86400) / 3600))
minutes=$(((${total_time} % 3600) / 60))

max_gpu_memory=$(cat ${gpu_mem_file})
max_ram_memory=$(cat ${ram_mem_file})

# GPU 및 RAM 모니터링 종료
kill $monitor_pid  # 백그라운드에서 실행 중인 모니터링 종료

# 최대 GPU 및 RAM 사용량을 로그 파일에 저장
echo "" >> ${loss_log_dir}/max_memory_usage.log
echo "${model} ${epoch} epochs, ${days} days ${hours} hours ${minutes} minutes" >> ${loss_log_dir}/max_memory_usage.log
echo "Start Time: ${start_datetime}, End Time: ${end_datetime}" >> ${loss_log_dir}/max_memory_usage.log
echo "Max GPU Memory Usage: ${max_gpu_memory} MiB, Max RAM Usage: ${max_ram_memory} MiB" >> ${loss_log_dir}/max_memory_usage.log

# 임시 파일 삭제
rm -f ${gpu_mem_file} ${ram_mem_file}
