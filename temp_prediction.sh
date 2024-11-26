#!/bin/bash

# <valid loss 낮은 epoch의 weight으로 prediction>

ckpt_DFHiC="/project/HiHiC/checkpoints_DFHiC/00024_0.10.39_0.0006255576.npz"
ckpt_DeepHiC="/project/HiHiC/checkpoints_DeepHiC/00104_3.46.21_0.0011968986"
ckpt_HiCARN1="/project/HiHiC/checkpoints_HiCARN1/00008_0.09.13_0.0009015936"
ckpt_HiCARN2="/project/HiHiC/checkpoints_HiCARN2/00010_0.19.04_0.0009090670"
ckpt_HiCNN2="/project/HiHiC/checkpoints_HiCNN/00021_0.51.09_0.0009432364"
ckpt_SRHiC="/project/HiHiC/checkpoints_SRHiC/00238_0.54.40_0.0007267343-496881.meta"
# ckpt_iEnhance="/project/HiHiC/checkpoints_iEnhance/00024_13.47.29_0.0000284136"
ckpt_iEnhance="/project/HiHiC/checkpoints_iEnhance/00282_6 days, 8.56.43_0.0000446402"
ckpt_HiCPlus="/project/HiHiC/checkpoints_HiCPlus/01000_0.39.06_0.0041882079"

models=("iEnhance")  # tensorflow: "SRHiC" "DFHiC" / torch: "HiCARN1" "HiCARN2" "HiCNN2" "DeepHiC" "iEnhance" "HiCPlus"
reads=("5000000" "4000000" "3000000" "2000000")

for model in "${models[@]}"; do
    ckpt_var="ckpt_${model}"
    checkpoint="${!ckpt_var}"  # Dynamic variable reference
    
    if [ ${model} == "SRHiC" ]; then # input type .npy
        for read in "${reads[@]}"; do
            bash model_prediction.sh -m ${model} -c ${checkpoint} -b 16 -g 0 -r ${read} -i ./data_${model}/for_enhancement_${model}_ch12-lx_${read}_10000.npy -o ./predicted_ch12-lx
            bash model_prediction.sh -m ${model} -c ${checkpoint} -b 16 -g 0 -r ${read} -i ./data_${model}/for_enhancement_${model}_GM12878_${read}_10000.npy -o ./predicted_GM12878
            bash model_prediction.sh -m ${model} -c ${checkpoint} -b 16 -g 0 -r ${read} -i ./data_${model}/for_enhancement_${model}_K562_${read}_10000.npy -o ./predicted_K562
        done
    
    elif [ ${model} == "DFHiC" ]; then # gpu 사용시 out of memory cpu => 배치단위 처리로 코드 수정
        for read in "${reads[@]}"; do
            bash model_prediction.sh -m ${model} -c ${checkpoint} -b 16 -g 0 -r ${read} -i ./data_${model}/for_enhancement_${model}_ch12-lx_${read}_10000.npz -o ./predicted_ch12-lx
            bash model_prediction.sh -m ${model} -c ${checkpoint} -b 16 -g 0 -r ${read} -i ./data_${model}/for_enhancement_${model}_GM12878_${read}_10000.npz -o ./predicted_GM12878
            bash model_prediction.sh -m ${model} -c ${checkpoint} -b 16 -g 0 -r ${read} -i ./data_${model}/for_enhancement_${model}_K562_${read}_10000.npz -o ./predicted_K562
        done

    elif [ "${model}" == "HiCARN1" ] || [ "${model}" == "HiCARN2" ]; then # 모델명과 데이터 폴더명 차이
        for read in "${reads[@]}"; do
            bash model_prediction.sh -m ${model} -c ${checkpoint} -b 16 -g 0 -r ${read} -i ./data_HiCARN/for_enhancement_HiCARN_ch12-lx_${read}_10000.npz -o ./predicted_ch12-lx
            bash model_prediction.sh -m ${model} -c ${checkpoint} -b 16 -g 0 -r ${read} -i ./data_HiCARN/for_enhancement_HiCARN_GM12878_${read}_10000.npz -o ./predicted_GM12878
            bash model_prediction.sh -m ${model} -c ${checkpoint} -b 16 -g 0 -r ${read} -i ./data_HiCARN/for_enhancement_HiCARN_K562_${read}_10000.npz -o ./predicted_K562
        done

    elif [ ${model} == "HiCNN2" ]; then # 모델명과 데이터 폴더명 차이
        for read in "${reads[@]}"; do
            bash model_prediction.sh -m ${model} -c ${checkpoint} -b 16 -g 0 -r ${read} -i ./data_HiCNN/for_enhancement_HiCNN_ch12-lx_${read}_10000.npz -o ./predicted_ch12-lx
            bash model_prediction.sh -m ${model} -c ${checkpoint} -b 16 -g 0 -r ${read} -i ./data_HiCNN/for_enhancement_HiCNN_GM12878_${read}_10000.npz -o ./predicted_GM12878
            bash model_prediction.sh -m ${model} -c ${checkpoint} -b 16 -g 0 -r ${read} -i ./data_HiCNN/for_enhancement_HiCNN_K562_${read}_10000.npz -o ./predicted_K562
        done

    elif [ ${model} == "iEnhance" ]; then # data_make_whole.py 안함
	    cell_types=("ch12-lx" "GM12878" "K562")
        for read in "${reads[@]}"; do
	        for cell_type in "${cell_types[@]}"; do	
                bash model_prediction.sh -m ${model} -c "${checkpoint}" -b 16 -g 0 -r ${read} -i ./data_${model}/for_enhancement_${cell_type}_${read}_10000.npz -o ./predicted_${model} -e ${cell_type}_ 
       	    done
	    done

    elif [ "${model}" == "DeepHiC" ] || [ "${model}" == "HiCPlus" ]; then
        for read in "${reads[@]}"; do
            bash model_prediction.sh -m ${model} -c ${checkpoint} -b 16 -g 0 -r ${read} -i ./data_${model}/for_enhancement_${model}_ch12-lx_${read}_10000.npz -o ./predicted_ch12-lx
            bash model_prediction.sh -m ${model} -c ${checkpoint} -b 16 -g 0 -r ${read} -i ./data_${model}/for_enhancement_${model}_GM12878_${read}_10000.npz -o ./predicted_GM12878
            bash model_prediction.sh -m ${model} -c ${checkpoint} -b 16 -g 0 -r ${read} -i ./data_${model}/for_enhancement_${model}_K562_${read}_10000.npz -o ./predicted_K562
        done
    fi 
done


# <prediction output submatrix를 intra chromosome 으로 통합>

dirs=("./predicted_ch12-lx" "./predicted_GM12878" "./predicted_K562")

for dir in "${dirs[@]}"; do
    for file in "${dir}"/*; do
        oldIFS=$IFS
        IFS='_' read -ra F <<< "$(basename "$file")"
        IFS=$oldIFS
        IFS='_' read -ra D <<< "$dir"
        # iEnhance가 아닌 경우에만 실행
        if [[ ${F[0]} != "iEnhance" ]]; then
            # Use $dir/$file for the correct relative path
            python data_make_whole.py -m ${F[0]} -i "${file}" -o ./predicted_${F[0]} -e ${D[-1]}_
        fi
    done
done
