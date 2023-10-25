#!/bin/bash

path=$(pwd)

# 텍스트 파일 경로 설정
file_path=${path}/GSM1551550_HIC001_merged_nodups.txt.gz

# 저장할 샘플링 파일 경로 설정
output_file=${path}/GSM1551550_HIC001_ds_16.txt.gz

# 파일의 총 라인 수 가져오기
# total_lines=$(wc -l < <(zcat "$file_path"))
total_lines=$(zcat "$file_path" | wc -l)
echo ${total_lines}

# 1/16을 샘플링할 라인 수 계산
sample_size=`expr ${total_lines} / 16`
echo ${sample_size}

# 중복 없이 무작위로 샘플링한 라인을 저장할 임시 파일 생성
temp_file=$(mktemp)

# 원본 파일에서 중복 없이 무작위로 샘플링한 라인을 추출하여 임시 파일에 저장
zcat "$file_path" | shuf -n "$sample_size" >> "$temp_file"

# 3번째와 7번째 컬럼을 기준으로 오름차순 정렬
sort -t"\t" -k3,3n -k7,7n <(gzip -c "temp_file") > "$output_file"

rm "$temp_file"  # 임시 파일 삭제

echo "중복 없이 샘플링된 라인이 $output_file 에 저장되었습니다."
