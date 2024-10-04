#!/bin/bash
seed=42
root_dir=$(pwd)

# GPU 및 RAM 최대 사용량 기록을 위한 변수 초기화
max_gpu_memory=0
max_ram_memory=0

# GPU 및 RAM 메모리 사용량을 기록하는 함수
monitor_resources() {
    while true; do
        # 현재 GPU 메모리 사용량 추출
        gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        
        # 현재 RAM 사용량 추출
        ram_memory=$(free -m | awk '/Mem:/ {print $3}')

        # 최대 GPU 메모리 사용량 계산
        if (( gpu_memory > max_gpu_memory )); then
            max_gpu_memory=$gpu_memory
        fi

        # 최대 RAM 사용량 계산
        if (( ram_memory > max_ram_memory )); then
            max_ram_memory=$ram_memory
        fi

        sleep 5  # 5초마다 체크
    done
}

# 백그라운드에서 GPU 및 RAM 모니터링 시작
monitor_resources &

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
echo "      After training, the models will be saved in ${output_model_dir}."
echo "      Loss variations and memory usage will be saved in ${loss_log_dir}."
echo "      If a GPU is available, it will be utilized - GPU:${gpu_id}."
echo ""

# validation set 유효성 검사 (hicplus 제외)
if [ -z "${valid_data_dir}" ] && [ "${model}" != "hicplus" ]; then
    echo "Error: Validation data directory is required for ${model} model." >&2
    exit 1
fi

mkdir -p ${loss_log_dir}

# 학습 시작 시간 기록
start_time=$(date +%s)

# 모델에 따라 학습 실행
case ${model} in
    hicplus)
        python model_hicplus/hicplus/train_models.py --root_dir ${root_dir} --model ${model} --epoch ${epoch} --batch_size ${batch_size} --gpu_id ${gpu_id} --output_model_dir ${output_model_dir} --loss_log_dir ${loss_log_dir} --train_data_dir ${train_data_dir}
        ;;
    HiCNN2)
        python model_HiCNN2/HiCNN2_training.py --root_dir ${root_dir} --model ${model} --epoch ${epoch} --batch_size ${batch_size} --gpu_id ${gpu_id} --output_model_dir ${output_model_dir} --loss_log_dir ${loss_log_dir} --train_data_dir ${train_data_dir} --valid_data_dir ${valid_data_dir}
        ;;
    SRHiC)
        python model_SRHiC/src/SRHiC_main.py --root_dir ${root_dir} --model ${model} --epoch ${epoch} --batch_size ${batch_size} --gpu_id ${gpu_id} --output_model_dir ${output_model_dir} --loss_log_dir ${loss_log_dir} --train_data_dir ${train_data_dir} --valid_data_dir ${valid_data_dir} --train True
        ;;
    DeepHiC)
        python model_DeepHiC/train.py --root_dir ${root_dir} --model ${model} --epoch ${epoch} --batch_size ${batch_size} --gpu_id ${gpu_id} --output_model_dir ${output_model_dir} --loss_log_dir ${loss_log_dir} --train_data_dir ${train_data_dir} --valid_data_dir ${valid_data_dir}
        ;;
    HiCARN1)
        python model_HiCARN/HiCARN_1_Train.py --root_dir ${root_dir} --model ${model} --epoch ${epoch} --batch_size ${batch_size} --gpu_id ${gpu_id} --output_model_dir ${output_model_dir} --loss_log_dir ${loss_log_dir} --train_data_dir ${train_data_dir} --valid_data_dir ${valid_data_dir}    
        ;;
    HiCARN2)
        python model_HiCARN/HiCARN_2_Train.py --root_dir ${root_dir} --model ${model} --epoch ${epoch} --batch_size ${batch_size} --gpu_id ${gpu_id} --output_model_dir ${output_model_dir} --loss_log_dir ${loss_log_dir} --train_data_dir ${train_data_dir} --valid_data_dir ${valid_data_dir}
        ;;
    DFHiC)
        python model_DFHiC/run_train.py --root_dir ${root_dir} --model ${model} --epoch ${epoch} --batch_size ${batch_size} --gpu_id ${gpu_id} --output_model_dir ${output_model_dir} --loss_log_dir ${loss_log_dir} --train_data_dir ${train_data_dir} --valid_data_dir ${valid_data_dir}
        ;;
    iEnhance)
        python model_iEnhance/train.py --train_option 0 --root_dir ${root_dir} --model ${model} --epoch ${epoch} --batch_size ${batch_size} --gpu_id ${gpu_id} --output_model_dir ${output_model_dir} --loss_log_dir ${loss_log_dir} --train_data_dir ${train_data_dir} --valid_data_dir ${valid_data_dir}
        ;;
    *)
        echo "Model name should be one of: hicplus, HiCNN2, SRHiC, DeepHiC, HiCARN1, HiCARN2, DFHiC, iEnhance." >&2
        exit 1
        ;;
esac

# 학습 종료 시간 기록
end_time=$(date +%s)

# 총 학습 시간 계산
total_time=$(($end_time - $start_time))
hours=$(($total_time / 3600))
minutes=$((($total_time % 3600) / 60))

# GPU 및 RAM 모니터링 종료
kill %1

# 최대 GPU 및 RAM 사용량을 로그 파일에 저장
echo "" >> ${loss_log_dir}/max_memory_usage.log
echo "${model} ${epoch} epochs, ${hours} hours ${minutes} minutes" >> ${loss_log_dir}/max_memory_usage.log
echo "Max GPU Memory Usage: ${max_gpu_memory} MiB, Max RAM Usage: ${max_ram_memory} MiB" >> ${loss_log_dir}/max_memory_usage.log
