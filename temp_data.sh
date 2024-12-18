#!/bin/bash
normalization=KR
resolutions=("10000" "25000")
reads=("5000000" "4000000" "3000000" "2000000")
models=("SRHiC") # "SRHiC" "DFHiC" "DeepHiC" "HiCARN" "HiCNN" "HiCPlus" "iEnhance"
flags=("trial1" "trial2" "trial3" "trial4" "trial5")

# wget "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1551620&format=file&file=GSM1551620%5FHIC071%5Fmerged%5Fnodups%2Etxt%2Egz" -O "GSM1551620_HIC071.txt.gz"
# wget "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM1551643&format=file&file=GSM1551643%5FHIC098%5Fmerged%5Fnodups%2Etxt%2Egz" -O "GSM1551643_HIC098.txt.gz"
# for resolution in "${resolutions[@]}"; do
#     for flag in "${flags[@]}"; do
#         for read in "${reads[@]}"; do
#             # <input_file> <ref_genome> <downsample_read> <juicertools> <normalization> <resolution> <saved_in>
#             bash data_downsample.sh GSM1551550_HIC001.txt.gz GM12878_${flag} hg19.txt ${read} ./juicer_tools.jar ${normalization} ${resolution} ./data
#             bash data_downsample.sh GSM1551620_HIC071.txt.gz K562_${flag} hg19.txt ${read} ./juicer_tools.jar ${normalization} ${resolution} ./data
#             bash data_downsample.sh GSM1551643_HIC098.txt.gz CH12-LX_${flag} mm9.chrom.sizes ${read} ./juicer_tools.jar ${normalization} ${resolution} ./data

#             if (( read >= 1000000 )); then
#                 read_abbr=$(awk -v res="$read" 'BEGIN { printf "%.1fM", res/1000000 }')
#             elif (( read >= 1000 )); then
#                 read_abbr=$(awk -v res="$read" 'BEGIN { printf "%.1fK", res/1000 }')
#             else
#                 read_abbr=$read
#             fi

#             if (( resolution >= 1000000 )); then
#                 resolution_abbr=$(awk -v res="$resolution" 'BEGIN { printf "%dMb", res/1000000 }')
#             elif (( resolution >= 1000 )); then
#                 resolution_abbr=$(awk -v res="$resolution" 'BEGIN { printf "%dKb", res/1000 }')
#             else
#                 resolution_abbr=$resolution
#             fi

#             for model in "${models[@]}"; do
#                 bash data_generate_for_prediction.sh -i ./data/GM12878_${flag}/MAT/GM12878_${flag}__${read_abbr}_${resolution_abbr}_${normalization} -b ${resolution} -m ${model} -g ./hg19.txt -o ./data_model -s 300
#                 bash data_generate_for_prediction.sh -i ./data/K562_${flag}/MAT/K562_${flag}__${read_abbr}_${resolution_abbr}_${normalization} -b ${resolution} -m ${model} -g ./hg19.txt -o ./data_model -s 300
#                 bash data_generate_for_prediction.sh -i ./data/CH12-LX_${flag}/MAT/CH12-LX_${flag}__${read_abbr}_${resolution_abbr}_${normalization} -b ${resolution} -m ${model} -g ./mm9.chrom.sizes -o ./data_model -s 300 
#             done
#         done
#     done
# done
models=("DeepHiC") # downsampling target 매트릭스 만들기
for model in "${models[@]}"; do
    dir_input=($(find "./data_model/data_${model}/ENHANCEMENT" -type f -name "*.npz"))
    for input in "${dir_input[@]}"; do
        python data_make_whole.py -i ${input} -o ./data_model_out
    done
done


# # 원본 target 매트릭스 만들기
# bash data_downsample.sh -i "GSM1551550_HIC001.txt.gz" -p "GM12878" -g "hg19.txt" -r "-1" -j "./juicer_tools.jar" -n "${normalization}" -b "${resolution}" -o "./data"
# bash data_downsample.sh -i "GSM1551620_HIC071.txt.gz" -p "K562" -g "hg19.txt" -r "-1" -j "./juicer_tools.jar" -n "${normalization}" -b "${resolution}" -o "./data"
# bash data_downsample.sh -i "GSM1551643_HIC098.txt.gz"  -p "CH12-LX" -g "mm9.chrom.sizes" -r "-1" -j "./juicer_tools.jar" -n "${normalization}" -b "${resolution}" -o "./data"
# resolutions=("10000" "25000")
# for resolution in "${resolutions[@]}"; do
#     models=("DeepHiC") 
#     for model in "${models[@]}"; do
#         bash data_generate_for_prediction.sh -i ./data/data_GM12878/MAT/*/ -b "${resolution}" -m "${model}" -g ./hg19.txt -o ./data_model/ORIGIN -s 300 
#         bash data_generate_for_prediction.sh -i ./data/data_K562/MAT/*/ -b "${resolution}" -m "${model}" -g ./hg19.txt -o ./data_model/ORIGIN -s 300
#         bash data_generate_for_prediction.sh -i ./data/data_CH12-LX/MAT/*/ -b "${resolution}" -m "${model}" -g ./mm9.chrom.sizes -o ./data_model/ORIGIN -s 300
#         dir_input=($(find "./data_model/ORIGIN/data_${model}/ENHANCEMENT" -type f -name "*.npz"))
#         for input in "${dir_input[@]}"; do
#             python data_make_whole.py -i "${input}" -o "./data_model_out"
#         done
#     done
# done
