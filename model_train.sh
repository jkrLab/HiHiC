#!/bin/bash
seed=42

root_dir=$(pwd)
model=$1
epoch=$2
batch_size=$3
gpu_id=$4
output_model_dir=$5
loss_log_dir=$6
train_data_dir=$7
if [ -z {$8} ]; then
valid_data_dir='specified directory, exept hicplus'
else
valid_data_dir=$8
fi

echo ""
echo "  ...Current working directory is ${root_dir}."
echo "      ${model} will train for ${epoch} epochs"
echo "      Training data is in ${train_data_dir},"
echo "      and validation data will be used in ${valid_data_dir}." 
echo "      After training, the models will be saved in ${output_model_dir}"
echo "      and loss trends will be saved in ${loss_log_dir} with epoch and time spent."
echo "      If possible, the training will perform using GPU:${gpu_id}."
echo ""


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
    python model_SRHiC/src/SRHiC_main.py --root_dir ${root_dir} --model ${model} --epoch ${epoch} --batch_size ${batch_size} --gpu_id ${gpu_id} --output_model_dir ${output_model_dir} --loss_log_dir ${loss_log_dir} --train_data_dir ${train_data_dir} --valid_data_dir ${valid_data_dir}

elif [ ${model} = "hicplus" ]; then
    python model_hicplus/hicplus/train_model.py --root_dir ${root_dir} --model ${model} --epoch ${epoch} --batch_size ${batch_size} --gpu_id ${gpu_id} --output_model_dir ${output_model_dir} --loss_log_dir ${loss_log_dir} --train_data_dir ${train_data_dir} --valid_data_dir ${valid_data_dir}

else
    echo "Model name should be one of the DeepHiC, HiCNN2, DFHiC, hicplus, HiCARN1, HiCARN2, and SRHiC."

fi