#!/bin/bash
normalization=KR
resolutions=("10000" "25000")
reads=("5000000" "4000000" "3000000" "2000000")
models=("iEnhance") # "SRHiC" "DFHiC" "DeepHiC" "HiCARN" "HiCNN" "HiCPlus" "iEnhance"
flags=("trial1" "trial2" "trial3" "trial4" "trial5")

# wget "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1551620&format=file&file=GSM1551620%5FHIC071%5Fmerged%5Fnodups%2Etxt%2Egz" -O "GSM1551620_HIC071.txt.gz"
# wget "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1551643&format=file&file=GSM1551643%5FHIC098%5Fmerged%5Fnodups%2Etxt%2Egz" -O "GSM1551643_HIC098.txt.gz"
for resolution in "${resolutions[@]}"; do
    for flag in "${flags[@]}"; do
        for read in "${reads[@]}"; do
            # <input_file> <ref_genome> <downsample_read> <juicertools> <normalization> <resolution> <saved_in>
            # bash data_downsample.sh GSM1551550_HIC001.txt.gz GM12878_${flag} hg19.txt ${read} ./juicer_tools.jar ${normalization} ${resolution} ./data
            # bash data_downsample.sh GSM1551620_HIC071.txt.gz K562_${flag} hg19.txt ${read} ./juicer_tools.jar ${normalization} ${resolution} ./data
            # bash data_downsample.sh GSM1551643_HIC098.txt.gz CH12-LX_${flag} mm9.chrom.sizes ${read} ./juicer_tools.jar ${normalization} ${resolution} ./data

            if (( read >= 1000000 )); then
                read_abbr=$(awk -v res="$read" 'BEGIN { printf "%.1fM", res/1000000 }')
            elif (( read >= 1000 )); then
                read_abbr=$(awk -v res="$read" 'BEGIN { printf "%.1fK", res/1000 }')
            else
                read_abbr=$read
            fi

           if (( resolution >= 1000000 )); then
                resolution_abbr=$(awk -v res="$resolution" 'BEGIN { printf "%dMb", res/1000000 }')
            elif (( resolution >= 1000 )); then
                resolution_abbr=$(awk -v res="$resolution" 'BEGIN { printf "%dKb", res/1000 }')
            else
                resolution_abbr=$resolution
            fi

            for model in "${models[@]}"; do
                bash data_generate_for_prediction.sh -i ./data/GM12878_${flag}/MAT/GM12878_${flag}__${read_abbr}_${resolution_abbr}_${normalization} -b ${resolution} -m ${model} -g ./hg19.txt -o ./data_model -s 300
                bash data_generate_for_prediction.sh -i ./data/K562_${flag}/MAT/K562_${flag}__${read_abbr}_${resolution_abbr}_${normalization} -b ${resolution} -m ${model} -g ./hg19.txt -o ./data_model -s 300
                bash data_generate_for_prediction.sh -i ./data/CH12-LX_${flag}/MAT/CH12-LX_${flag}__${read_abbr}_${resolution_abbr}_${normalization} -b ${resolution} -m ${model} -g ./mm9.chrom.sizes -o ./data_model -s 300 
            done
        done

        # 확인용 target 매트릭스 만들기
        # for read in "${reads[@]}"; do
        #     models=("iEnhance" "DeepHiC")
        #     for model in "${models[@]}"; do
        #         bash data_generate_for_prediction.sh -i ./data_GM12878_ds_${read}/resolution${resolution}${flag} -b ${resolution} -m ${model} -g ./hg19.txt -o ./data_downsample -s 300 -n ${normalization} -e _GM12878_${read}${flag}
        #         bash data_generate_for_prediction.sh -i ./data_K562_ds_${read}/resolution${resolution}${flag} -b ${resolution} -m ${model} -g ./hg19.txt -o ./data_downsample -s 300 -n ${normalization} -e _K562_${read}${flag}
        #         bash data_generate_for_prediction.sh -i ./data_ch12-lx_ds_${read}/resolution${resolution}${flag} -b ${resolution} -m ${model} -g ./mm9.chrom.sizes -o ./data_downsample -s 300 -n ${normalization} -e _ch12-lx_${read}${flag}
        #     done
        # done
        # # submatrix를 크로모좀 별로 어셈블
        # dirs="./data_downsample"
        # for dir in "${dirs}"/; do
        #     for file in "${dir}"/*; do
        #         oldIFS=$IFS
        #         IFS='_' read -ra F <<< "$(basename "$file")"
        #         IFS=$oldIFS
        #         IFS='_' read -ra D <<< "$dir"
        #         # iEnhance가 아닌 경우에만 실행
        #         if [[ ${F[0]} != "iEnhance" ]]; then
        #             # Use $dir/$file for the correct relative path
        #             python data_make_whole.py -m ${F[-1]} -i "${file}" -o ./targeted_data_downsample
        #         fi
        #     done
        # done
        # # 파일 이름 변경
        # target_dir="./targeted_data_downsample"
        # for file in "${target_dir}"/*.npz; do
        #     base=$(basename "$file")
        #     new_name=$(echo "$base" | sed 's/for_enhancement_[^_]*_//')
        #     if [[ "$base" != "$new_name" ]]; then
        #         mv "$file" "${target_dir}/${new_name}"
        #     fi
        # done
    done
done
# # 다운 sampling 하지 않은 원본 매트릭스 만들기
# bash data_downsample.sh GSM1551550_HIC001 hg19 162996536 ./juicer_tools.jar ${normalization} ${resolution} data_GM12878
# bash data_downsample.sh GSM1551620_HIC071 hg19 71129305 ./juicer_tools.jar ${normalization} ${resolution} data_K562
# bash data_downsample.sh GSM1551643_HIC098 mm9.chrom.sizes 113110795 ./juicer_tools.jar ${normalization} ${resolution} data_ch12-lx
# models=("iEnhance" "DeepHiC")
# for model in "${models[@]}"; do
#     bash data_generate_for_prediction.sh -i /project/HiHiC/data_GM12878_downsampled_162996536 -b ${resolution} -m ${model} -g ./hg19.txt -o ./data_target -s 300 -n ${normalization} -e _GM12878_162996536
#     bash data_generate_for_prediction.sh -i /project/HiHiC/data_K562_downsampled_71129305 -b ${resolution} -m ${model} -g ./hg19.txt -o ./data_target -s 300 -n ${normalization} -e _K562_71129305
#     bash data_generate_for_prediction.sh -i /project/HiHiC/data_ch12-lx_downsampled_113110795 -b ${resolution} -m ${model} -g ./mm9.chrom.sizes -o ./data_target -s 300 -n ${normalization} -e _ch12-lx_113110795
# done
# # submatrix를 크로모좀 별로 어셈블
# dirs="./data_target"
# for dir in "${dirs}"/; do
#     for file in "${dir}"/*; do
#         oldIFS=$IFS
#         IFS='_' read -ra F <<< "$(basename "$file")"
#         IFS=$oldIFS
#         IFS='_' read -ra D <<< "$dir"
#         # iEnhance가 아닌 경우에만 실행
#         if [[ ${F[0]} != "iEnhance" ]]; then
#             # Use $dir/$file for the correct relative path
#             python data_make_whole.py -m ${D[-1]} -i "${file}" -o ./targeted_data
#         fi
#     done
# done
# 파일 이름 변경
# target_dir="/project/HiHiC/data_target"
# for file in "${target_dir}"/*.npz; do
#     base=$(basename "$file")
#     new_name=$(echo "$base" | sed 's/for_enhancement_[^_]*_//')

#     if [[ "$base" != "$new_name" ]]; then
#         mv "$file" "${target_dir}/${new_name}"
#     fi
# done