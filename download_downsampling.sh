#!/bin/bash

path=$(pwd)

file_url="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551550/suppl/GSM1551550_HIC001_merged_nodups.txt.gz"
file_name="GSM1551550_HIC001.txt.gz"



### read data download ###
wget ${file_url} -O ${file_name} 


### downsampling ###
# 텍스트 파일 경로 설정
file_path=${path}/${file_name}

# 저장할 샘플링 파일 경로 설정
output_file=${path}/GSM1551550_HIC001_ds_16.txt.gz

# 파일의 총 라인 수 가져오기
total_lines=$(zcat "$file_path" | wc -l)
echo "The number of total reads is ${total_lines}."

# 1/16을 샘플링할 라인 수 계산
sample_size=`expr ${total_lines} / 16`

# 중복 없이 무작위로 샘플링한 라인을 저장할 임시 파일 생성
temp_file=$(mktemp)

# 원본 파일에서 중복 없이 무작위로 샘플링한 라인을 추출하여 임시 파일에 저장
zcat "$file_path" | shuf -n "$sample_size" >> "$temp_file"
echo "And ${sample_size} reads are randomly sampled."

# 3번째와 7번째 컬럼(크로모좀) 기준으로 오름차순 정렬
sort -k3,3d -k7,7 "$temp_file" | gzip > "$output_file"
rm "$temp_file"  # 임시 파일 삭제
echo "중복 없이 샘플링된 reads가 $output_file 에 저장되었습니다."


### .hic format ###
# juicer pre 로 .hic 변환
java -Xmx2g -jar ../juicer_tools.2.20.00.jar pre ${file_name} ./GSM1551550.hic hg19 # 원본 데이터
java -Xmx2g -jar ../juicer_tools.2.20.00.jar pre ${output_file} ./GSM1551550_down16.hic hg19 # 다운 샘플링 데이터


### .txt format for each chromosome (intra) ###
# 크로모좀의 시작과 끝 숫자 설정
start=1
end=23

# 결과를 저장할 폴더 생성​​
mkdir ${path}/data
# 전체 크로모좀에 대하여 반복문으로 명령 실행
for ((chrom=1; chrom<end; chrom++))
do
    java -jar ../juicer_tools.2.20.00.jar dump observed NONE ./GSM1551550.hic ${chrom} ${chrom} BP 10000 ${path}/data/chr${chrom}_10kb.txt
done
echo "원본데이터의 매트릭스 정보가 ./data/ 에 저장되었습니다."

# 결과를 저장할 폴더 생성​​
mkdir ${path}/downsampled_data
# 전체 크로모좀에 대하여 반복문으로 명령 실행
for ((chrom=1; chrom<end; chrom++))
do
    java -jar ../juicer_tools.2.20.00.jar dump observed NONE ./GSM1551550_down16.hic ${chrom} ${chrom} BP 10000 ${path}/downsampled_data/chr${chrom}_10kb.txt
done
echo "다운 샘플링된 데이터의 매트릭스 정보가 downsampled_data/ 에 저장되었습니다."