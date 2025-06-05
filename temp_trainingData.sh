#!/bin/bash

reads=("10000000")
resolution=10000
models=("SRHiC")  # tensorflow: "SRHiC" "DFHiC" / torch: "HiCARN1" "HiCARN2" "HiCNN2" "DeepHiC" "iEnhance" "HiCPlus"

# bash data_download_downsample.sh -i "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551550/suppl/GSM1551550_HIC001_merged_nodups.txt.gz" -p "GM12878" -g "hg19.txt" -r "${read}" -j "./juicer_tools.jar" -n "KR" -b ${resolution} -o "./data"
# bash data_downsample.sh -i "GSM1551550_HIC001.txt.gz" -p "GM12878" -g "hg19.txt" -r "3000000" -j "juicer_tools.jar" -n "KR" -b "${resolution}" -o "./data"

for read in "${reads[@]}"; do
#     # <input_file> <ref_genome> <downsample_read> <juicertools> <normalization> <resolution> <saved_in>

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
        echo ${model}
        bash data_generate_for_training.sh -i "./data/GM12878/MAT/GM12878__163.0M_${resolution_abbr}_KR" -d "./data/GM12878/MAT/GM12878__${read_abbr}_${resolution_abbr}_KR" -r "${read}" -b "${resolution}" -m "${model}" -g "./hg19.txt" -o "./data_model" -s "300" -t "1 2 3 4 5 6 7 8 9 10 11 12 13 14" -v "15 16 17" -p "18 19 20 21 22"
    done
done
