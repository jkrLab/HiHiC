#!/bin/bash

# scp -r ./targeted_data/ mohyelim7@203.253.2.168:/home/data/HiHiC
# scp -r ./targeted_data_downsample/ mohyelim7@203.253.2.168:/home/data/HiHiC

cells=("GM12878" "K562" "CH12-LX")
reads=("2.0M" "3.0M" "4.0M" "5.0M")
trials=("trial1" "trial2" "trial3" "trial4" "trial5")

for trial in "${trials[@]}"; do
    for read in "${reads[@]}"; do
        for cell in "${cells[@]}"; do
            scp ./data_model_out/${cell}_${trial}/${cell}_${trial}__${read}* mohyelim7@203.253.2.168:/home/data/HiHiC/${cell}__${read}/
        done
    done
done