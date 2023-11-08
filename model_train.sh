#!/bin/bash
seed=42

path=$(pwd)
model=$1
epoch=$2
gpu_id=$3
output_model_dir=$4
loss_log_dir=$5
train_data_dir=$6
if [ -z ${$7} ]; then
valid_data_dir='specified directory, exept hicplus'
else
valid_data_dir=$7
fi

echo ""
echo "  ...Current working directory is ${path}."
echo "      ${model} will train for ${epoch} epochs"
echo "      Training data is in ${train_data_dir},"
echo "      and validation data will be used in ${valid_data_dir}." 
echo "      After training, the models will be saved in ${output_model_dir}"
echo "      and loss trends will be saved in ${loss_log_dir} with epoch and time spent."
echo "      If possible, the training will perform using GPU:${gpu_id}."
echo ""


if [ ${model}=HiCARN1 ]; then
python model_HiCARN/HiCARN_1_Train.py -m ${model} -t ${train_data_dir} -v ${valid_data_dir} -o ${output_model_dir} -e ${epoch} -g ${gpu_id} -l ${loss_log_dir}

if [ ${model}=HiCARN2 ]; then
python model_HiCARN/HiCARN_2_Train.py -m ${model} -t ${train_data_dir} -v ${valid_data_dir} -o ${output_model_dir} -e ${epoch} -g ${gpu_id} -l ${loss_log_dir}

elif [ ${model}=DeepHiC ]; then
python model_DeepHiC/train.py -m ${model} -t ${train_data_dir} -v ${valid_data_dir} -o ${output_model_dir} -e ${epoch} -g ${gpu_id} -l ${loss_log_dir}

elif [ ${model}=HiCNN2 ]; then
python model_HiCNN2/HiCNN2_training.py -m ${model} -t ${train_data_dir} -v ${valid_data_dir} -o ${output_model_dir} -e ${epoch} -g ${gpu_id} -l ${loss_log_dir}

elif [ ${model}=DFHiC ]; then
python model_DFHiC/run_train.py -m ${model} -t ${train_data_dir} -v ${valid_data_dir} -o ${output_model_dir} -e ${epoch} -g ${gpu_id} -l ${loss_log_dir}

elif [ ${model}=SRHiC ]; then
python src/SRHiC_main.py -m ${model} -t ${train_data_dir} -v ${valid_data_dir} -o ${output_model_dir} -e ${epoch} -g ${gpu_id} -l ${loss_log_dir}

elif [ ${model}=hicplus ]; then
python model_hicplus/hicplus/train_model.py -m ${model} -t ${train_data_dir} -o ${output_model_dir} -e ${epoch} -g ${gpu_id} -l ${loss_log_dir}

else
echo "Model name should be one of the DeepHiC, HiCNN2, DFHiC, hicplus, HiCARN1, HiCARN2, and SRHiC."

fi