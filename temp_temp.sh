path=$(pwd)

# resolution=25000
# flags=("" "_resample_1" "_resample_2" "_resample_3" "_resample_4")
# reads=("2000000" "3000000" "4000000" "5000000")
# for read in "${reads[@]}"; do
#   for flag in "${flags[@]}"; do
#     mkdir -p "./data_ch12-lx_downsampled_${read}${flag}/res${resolution}/"
#     for ((chrom=1; chrom<20; chrom++)); do
#       java -jar ./juicer_tools.jar dump observed KR ./GSM1551643_HIC098_ds_${read}${flag}.hic \
#           "$chrom" "$chrom" BP ${resolution} ./data_ch12-lx_downsampled_${read}${flag}/res${resolution}/chr${chrom}_${resolution}.txt
#     done
#     mkdir -p "./data_GM12878_downsampled_${read}${flag}/res${resolution}/"
#     mkdir -p "./data_K562_downsampled_${read}${flag}/res${resolution}/"
#     for ((chrom=1; chrom<23; chrom++)); do
#       java -jar ./juicer_tools.jar dump observed KR ./GSM1551550_HIC001_ds_${read}${flag}.hic \
#       "$chrom" "$chrom" BP ${resolution} ./data_GM12878_downsampled_${read}${flag}/res${resolution}/chr${chrom}_${resolution}.txt
#       java -jar ./juicer_tools.jar dump observed KR ./GSM1551620_HIC071_ds_${read}${flag}.hic \
#       "$chrom" "$chrom" BP ${resolution} ./data_K562_downsampled_${read}${flag}/res${resolution}/chr${chrom}_${resolution}.txt
#     done
#   done
# done

# for model in "${models[@]}"; do
#   for read in "${reads[@]}"; do
#     for flag in "${flags[@]}"; do
#       # -i <input_data_path> -b <bin_size> -m <model_name> -g <ref_chromosome_length> -o <output_path> -s <max_value> [-n <normalization>]
#       bash data_generate_for_prediction.sh -i /project/HiHiC/data_GM12878_downsampled_${read}${flag}/res${resolution} -b ${resolution} -m ${model} -g ./hg19.txt -o ./ -s 300 -n ${normalization} -e _GM12878_${read}${flag}
#       bash data_generate_for_prediction.sh -i /project/HiHiC/data_K562_downsampled_${read}${flag}/res${resolution} -b ${resolution} -m ${model} -g ./hg19.txt -o ./ -s 300 -n ${normalization} -e _K562_${read}${flag}
#       bash data_generate_for_prediction.sh -i /project/HiHiC/data_ch12-lx_downsampled_${read}${flag}/res${resolution} -b ${resolution} -m ${model} -g ./mm9.chrom.sizes -o ./ -s 300 -n ${normalization} -e _ch12-lx_${read}${flag}
#     done
#   done
# done

