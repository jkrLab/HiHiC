#!/bin/bash

path=$(pwd)
# input=${path}/GSM1551550_HIC001_merged_nodups.txt.gz
output=${path}/GSM1551550_subsample.txt
# gunzip ${input}
input=${path}/GSM1551550_HIC001_merged_nodups.txt

allline=$(cat "$input" | wc -l)
subsetline=`expr ${allline} / 16`

if [ -f ${output} ]; then
	rm ${output}
fi

shuf -i 1-${allline} -n ${subsetline} | sort -n | while read line
do
	sed -n "${line}p" ${input} >> ${output}
done