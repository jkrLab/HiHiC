#!/bin/bash

# scp -r /data/HiHiC/data_model_out/GM12878_3.0M mohyelim7@203.253.2.168:/home/data/HiHiC/for_loop_calling
# scp -r /data/HiHiC/data_model_out/GM12878_10.0M mohyelim7@203.253.2.168:/home/data/HiHiC/for_loop_calling

cells=("GM12878" "K562" "CH12-LX")
reads=("2.0M" "3.0M" "4.0M" "5.0M")
trials=("trial1" "trial2" "trial3" "trial4" "trial5")
bins=("10Kb" "25Kb")

# for cell in "${cells[@]}"; do
#     for read in "${reads[@]}"; do
#         for trial in "${trials[@]}"; do
#             # scp ./data_model_out/${cell}_${trial}/${cell}_${trial}__${read}* mohyelim7@203.253.2.168:/home/data/HiHiC/train_3.0M/${cell}__${read}/            
#         done
#     done
# done


for cell in "${cells[@]}"; do
    scp /data/HiHiC/data/${cell}/HIC/*.hic mohyelim7@203.253.2.168:/home/data/HiHiC/HIC/
    for read in "${reads[@]}"; do
        scp /data/HiHiC/data/${cell}_trial1/HIC/${cell}_trial1__${read}.hic mohyelim7@203.253.2.168:/home/data/HiHiC/HIC/    
    done
done