#!/bin/bash
seed=42
path=$(pwd)

download_url="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551550/suppl/GSM1551550_HIC001_merged_nodups.txt.gz"
file_name="GSM1551550_HIC001"
chromosom_length="hg19"
downsample_ratio="16"
juicertools="./juicer_tools.jar" # path to juicer_tool.jar of docker image


### read data download ###
save_name="${file_name}.txt.gz"
wget ${download_url} -O ${save_name} 


### downsampling ###
# 텍스트 파일 경로 설정
file_path=${path}/${save_name}

# 저장할 샘플링 파일 경로 설정
downsample_name=${path}/${file_name}_ds_${downsample_ratio}.txt.gz

# 파일의 총 라인 수 가져오기
total_lines=$(zcat "$file_path" | wc -l)
echo "The number of total reads is ${total_lines}."

# downsample_ratio로 샘플링할 라인 수 계산
sample_size=`expr ${total_lines} / ${downsample_ratio}`

# 중복 없이 무작위로 샘플링한 라인을 저장할 임시 파일 생성
temp_file=$(mktemp)

# 원본 파일에서 중복 없이 무작위로 샘플링한 라인을 추출하여 임시 파일에 저장
zcat "$file_path" | shuf -n "$sample_size" >> "$temp_file"
echo "And ${sample_size} reads are randomly sampled."

# 3번째와 7번째 컬럼(크로모좀) 기준으로 오름차순 정렬
sort -k3,3d -k7,7 "$temp_file" | gzip > "${downsample_name}"
rm "$temp_file"  # 임시 파일 삭제
echo "Randomly sampled reads are saved: ${downsample_name}"


### .hic format ###
# juicer pre 로 .hic 변환
java -Xmx2g -jar ${juicertools} pre ${save_name} ./${file_name}.hic ${chromosom_length} # 원본 데이터
echo "${file_name}.hic is generated."

java -Xmx2g -jar ${juicertools} pre ${downsample_name} ./${file_name}_ds_${downsample_ratio}.hic ${chromosom_length} # 다운 샘플링 데이터
echo "${file_name}_ds_${downsample_ratio}.hic is generated."

### .txt format for each chromosome (intra) ###
# 크로모좀의 시작과 끝 숫자 설정
start=1
end=23

# 결과를 저장할 폴더 생성​​
mkdir ${path}/data
# 전체 크로모좀에 대하여 반복문으로 명령 실행
for ((chrom=1; chrom<end; chrom++))
do
    java -jar ${juicertools} dump observed NONE ./${file_name}.hic ${chrom} ${chrom} BP 10000 ${path}/data/chr${chrom}_10kb.txt
done
echo "Intra chromosome contact matrix are generated: ./data/"

# 결과를 저장할 폴더 생성​​
mkdir ${path}/data_downsampled_${downsample_ratio}
# 전체 크로모좀에 대하여 반복문으로 명령 실행
for ((chrom=1; chrom<end; chrom++))
do
    java -jar ${juicertools} dump observed NONE ./${file_name}_ds_${downsample_ratio}.hic ${chrom} ${chrom} BP 10000 ${path}/data_downsampled_${downsample_ratio}/chr${chrom}_10kb.txt
done
echo "Downsampled contact matrix are generated: ./data_downsampled_${downsample_ratio}/"