#!/bin/bash
seed=42
root_dir=$(pwd)

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

root_dir=$(pwd)

echo ""
echo "  ...Current working directory is ${root_dir}."
echo "      ${model} will train for ${epoch} epochs"
echo "      Training data is in ${train_data_dir},"
echo "      and validation data will be used in ${valid_data_dir}." 
echo "      After training, the models will be saved in ${output_model_dir}"
echo "      and loss trends will be saved in ${loss_log_dir} with epoch and time spent."
echo "      If possible, the training will perform using GPU:${gpu_id}."
echo ""

if [ -z "${valid_data_dir}" ]; then
    valid_data_dir='specified valid data directory'
fi

if [ ${model} = "HiCARN1" ]; then
    python model_HiCARN/HiCARN_1_Train.py --root_dir ${root_dir} --model ${model} --epoch ${epoch} --batch_size ${batch_size} --gpu_id ${gpu_id} --output_model_dir ${output_model_dir} --loss_log_dir ${loss_log_dir} --train_data_dir ${train_data_dir} --valid_data_dir ${valid_data_dir}    

elif [ ${model} = "HiCARN2" ]; then
    python model_HiCARN/HiCARN_2_Train.py --root_dir ${root_dir} --model ${model} --epoch ${epoch} --batch_size ${batch_size} --gpu_id ${gpu_id} --output_model_dir ${output_model_dir} --loss_log_dir ${loss_log_dir} --train_data_dir ${train_data_dir} --valid_data_dir ${valid_data_dir}

elif [ ${model} = "DeepHiC" ]; then
    python model_DeepHiC/train.py --root_dir ${root_dir} --model ${model} --epoch ${epoch} --batch_size ${batch_size} --gpu_id ${gpu_id} --output_model_dir ${output_model_dir} --loss_log_dir ${loss_log_dir} --train_data_dir ${train_data_dir} --valid_data_dir ${valid_data_dir}

elif [ ${model} = "HiCNN2" ]; then
    python model_HiCNN2/HiCNN2_training.py --root_dir ${root_dir} --model ${model} --epoch ${epoch} --batch_size ${batch_size} --gpu_id ${gpu_id} --output_model_dir ${output_model_dir} --loss_log_dir ${loss_log_dir} --train_data_dir ${train_data_dir} --valid_data_dir ${valid_data_dir}

elif [ ${model} = "DFHiC" ]; then
    python model_DFHiC/run_train.py --root_dir ${root_dir} --model ${model} --epoch ${epoch} --batch_size ${batch_size} --gpu_id ${gpu_id} --output_model_dir ${output_model_dir} --loss_log_dir ${loss_log_dir} --train_data_dir ${train_data_dir} --valid_data_dir ${valid_data_dir}

elif [ ${model} = "SRHiC" ]; then
    python model_SRHiC/src/SRHiC_main.py --root_dir ${root_dir} --model ${model} --epoch ${epoch} --batch_size ${batch_size} --gpu_id ${gpu_id} --output_model_dir ${output_model_dir} --loss_log_dir ${loss_log_dir} --train_data_dir ${train_data_dir} --valid_data_dir ${valid_data_dir} --train True

elif [ ${model} = "hicplus" ]; then
    python model_hicplus/hicplus/train_models.py --root_dir ${root_dir} --model ${model} --epoch ${epoch} --batch_size ${batch_size} --gpu_id ${gpu_id} --output_model_dir ${output_model_dir} --loss_log_dir ${loss_log_dir} --train_data_dir ${train_data_dir}

else
    echo "Model name should be one of the DeepHiC, HiCNN2, DFHiC, hicplus, HiCARN1, HiCARN2, and SRHiC."

fi

mkdir -p ${loss_log_dir}