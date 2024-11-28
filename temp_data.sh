#!/bin/bash

reads=("5000000" "4000000" "3000000" "2000000")
for read in "${reads[@]}"; do
    # <file_prefix> <ref_genome> <downsample_read> <juicertools> <normalization> <resolution> <saved_in>
    bash data_downsample.sh GSM1551550_HIC001 hg19 ${read} ./juicer_tools.jar KR 10000 data_GM12878
    bash data_downsample.sh GSM1551620_HIC071 hg19 ${read} ./juicer_tools.jar KR 10000 data_K562
    bash data_downsample.sh GSM1551643_HIC098 mm9.chrom.sizes ${read} ./juicer_tools.jar KR 10000 data_ch12-lx
    models=("SRHiC" "DFHiC" "DeepHiC" "HiCARN" "HiCNN" "HiCPlus" "iEnhance")
    for model in "${models[@]}"; do
        # -i <input_data_path> -b <bin_size> -m <model_name> -g <ref_chromosome_length> -o <output_path> -s <max_value> [-n <normalization>]
        bash data_generate_for_prediction.sh -i /project/HiHiC/data_GM12878_downsampled_${read} -b 10000 -m ${model} -g ./hg19.txt -o ./ -s 300 -n KR -e _GM12878_${read}
        bash data_generate_for_prediction.sh -i /project/HiHiC/data_K562_downsampled_${read} -b 10000 -m ${model} -g ./hg19.txt -o ./ -s 300 -n KR -e _K562_${read}
        bash data_generate_for_prediction.sh -i /project/HiHiC/data_ch12-lx_downsampled_${read} -b 10000 -m ${model} -g ./mm9.chrom.sizes -o ./ -s 300 -n KR -e _ch12-lx_${read}
    done
done


# 다운 sampling 하지 않은 원본 매트릭스 만들기
bash data_downsample.sh GSM1551550_HIC001 hg19 162996536 ./juicer_tools.jar KR 10000 data_GM12878
bash data_downsample.sh GSM1551620_HIC071 hg19 71129305 ./juicer_tools.jar KR 10000 data_K562
bash data_downsample.sh GSM1551643_HIC098 mm9.chrom.sizes 113110795 ./juicer_tools.jar KR 10000 data_ch12-lx
models=("iEnhance" "DeepHiC")
for model in "${models[@]}"; do
    bash data_generate_for_prediction.sh -i /project/HiHiC/data_GM12878_downsampled_162996536 -b 10000 -m ${model} -g ./hg19.txt -o ./data_target -s 300 -n KR -e _GM12878_162996536
    bash data_generate_for_prediction.sh -i /project/HiHiC/data_K562_downsampled_71129305 -b 10000 -m ${model} -g ./hg19.txt -o ./data_target -s 300 -n KR -e _K562_71129305
    bash data_generate_for_prediction.sh -i /project/HiHiC/data_ch12-lx_downsampled_113110795 -b 10000 -m ${model} -g ./mm9.chrom.sizes -o ./data_target -s 300 -n KR -e _ch12-lx_113110795
done
# submatrix를 크로모좀 별로 어셈블
dirs="./data_target"
for dir in "${dirs}"/; do
    for file in "${dir}"/*; do
        oldIFS=$IFS
        IFS='_' read -ra F <<< "$(basename "$file")"
        IFS=$oldIFS
        IFS='_' read -ra D <<< "$dir"
        # iEnhance가 아닌 경우에만 실행
        if [[ ${F[0]} != "iEnhance" ]]; then
            # Use $dir/$file for the correct relative path
            python data_make_whole.py -m ${D[-1]} -i "${file}" -o ./targeted_data
        fi
    done
done
파일 이름 변경
target_dir="/project/HiHiC/data_target"
for file in "${target_dir}"/*.npz; do
    base=$(basename "$file")
    new_name=$(echo "$base" | sed 's/for_enhancement_[^_]*_//')

    if [[ "$base" != "$new_name" ]]; then
        mv "$file" "${target_dir}/${new_name}"
    fi
done

# 확인용 target 매트릭스 만들기
reads=("5000000" "4000000" "3000000" "2000000")
for read in "${reads[@]}"; do
    models=("iEnhance" "DeepHiC")
    for model in "${models[@]}"; do
        bash data_generate_for_prediction.sh -i /project/HiHiC/data_GM12878_downsampled_${read} -b 10000 -m ${model} -g ./hg19.txt -o ./data_downsample -s 300 -n KR -e _GM12878_${read}
        bash data_generate_for_prediction.sh -i /project/HiHiC/data_K562_downsampled_${read} -b 10000 -m ${model} -g ./hg19.txt -o ./data_downsample -s 300 -n KR -e _K562_${read}
        bash data_generate_for_prediction.sh -i /project/HiHiC/data_ch12-lx_downsampled_${read} -b 10000 -m ${model} -g ./mm9.chrom.sizes -o ./data_downsample -s 300 -n KR -e _ch12-lx_${read}
    done
done
# submatrix를 크로모좀 별로 어셈블
dirs="./data_downsample"
for dir in "${dirs}"/; do
    for file in "${dir}"/*; do
        oldIFS=$IFS
        IFS='_' read -ra F <<< "$(basename "$file")"
        IFS=$oldIFS
        IFS='_' read -ra D <<< "$dir"
        # iEnhance가 아닌 경우에만 실행
        if [[ ${F[0]} != "iEnhance" ]]; then
            # Use $dir/$file for the correct relative path
            python data_make_whole.py -m ${F[-1]} -i "${file}" -o ./targeted_data_downsample
        fi
    done
done
# 파일 이름 변경
target_dir="./targeted_data_downsample"
for file in "${target_dir}"/*.npz; do
    base=$(basename "$file")
    new_name=$(echo "$base" | sed 's/for_enhancement_[^_]*_//')
    if [[ "$base" != "$new_name" ]]; then
        mv "$file" "${target_dir}/${new_name}"
    fi
done