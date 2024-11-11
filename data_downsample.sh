#!/bin/bash
set -e  # Error handling: stop script on any error
set -u  # Treat unset variables as an error
seed=42

# examples
# bash data_downsample.sh GSM1551550_HIC001 hg19 5000000 ./juicer_tools.jar KR 10000 data_GM12878
# bash data_downsample.sh GSM1551620_HIC071 hg19 5000000 ./juicer_tools.jar KR 10000 data_K562
# bash data_downsample.sh GSM1551643_HIC098 mm9.chrom.sizes 5000000 ./juicer_tools.jar KR 10000 data_ch12-lx

# 필요한 매개변수 체크
if [ "$#" -ne 7 ]; then
  echo "Usage: $0 <file_prefix> <ref_genome> <downsample_read> <juicertools> <normalization> <resolution> <saved_in>"
  exit 1
fi

### you can modifiy here as string type with ""!!!
path=$(pwd)
file_name=$1
ref_genome=$2
downsample_read=$3
juicertools=$4
normalization=$5
resolution=$6
saved_in=$7

echo ""
echo "  ...current working directory is $(pwd)"
echo "      raw data file : ${file_name}"
echo "      refrence genome : ${ref_genome}"
echo "      downsampling read : ${downsample_read}"
echo "      juicertools : ${juicertools}" # path to juicer_tool.jar of docker image
echo "      normalization method : ${normalization}"
echo "      binning resolution : ${resolution}"
echo "      intra chromosome data : ${saved_in}/ and ${saved_in}_downsampled_${downsample_read}/"
echo ""

save_name="${file_name}.txt.gz"
file_path=${path}/${save_name}

# 저장할 샘플링 파일 경로 설정
downsample_name=${path}/${file_name}_ds_${downsample_read}.txt.gz

# 파일의 총 라인 수 가져오기
total_lines=$(zcat "$file_path" | wc -l)
echo ""
echo "  ...The number of total reads is ${total_lines}."

if [ "$downsample_read" -gt "$total_lines" ]; then
  echo "Error: downsample_read is greater than the total number of reads ($total_lines). Please check your input."
  exit 1
fi

# 중복 없이 무작위로 샘플링한 라인을 저장할 임시 파일 생성
temp_file=$(mktemp)

# 원본 파일에서 중복 없이 무작위로 샘플링한 라인을 추출하여 임시 파일에 저장
zcat "$file_path" | shuf -n "$downsample_read" >> "$temp_file"
echo "  ...And ${downsample_read} reads are randomly sampled."

# 3번째와 7번째 컬럼(크로모좀) 기준으로 오름차순 정렬
sort -k3,3d -k7,7 "$temp_file" | gzip > "${downsample_name}"
rm "$temp_file"  # 임시 파일 삭제
echo "  ...Randomly sampled reads are saved: ${downsample_name}"


### .hic format ###
# juicer pre 로 .hic 변환
echo ""
echo "  ...To hic,"
java -Xmx2g -jar ${juicertools} pre ${downsample_name} ./${file_name}_ds_${downsample_read}.hic ${ref_genome} # 다운 샘플링 데이터
echo "  ...${file_name}_ds_${downsample_read}.hic is generated."


### .txt format for each chromosome (intra) ###
# ref_genome에 따라 크로모좀 범위 설정
if [[ "$ref_genome" == hg* ]]; then
  start=1
  end=23
elif [[ "$ref_genome" == mm* ]]; then
  start=1
  end=20
else
  echo "Reference genome: ${ref_genome}: chr${start} - chr${end}"
  exit 1
fi


# 결과를 저장할 폴더 생성​​
mkdir ${path}/${saved_in}_downsampled_${downsample_read}
# 전체 크로모좀에 대하여 반복문으로 명령 실행
for ((chrom=1; chrom<end; chrom++))
do
    java -jar ${juicertools} dump observed ${normalization} ./${file_name}_ds_${downsample_read}.hic ${chrom} ${chrom} BP ${resolution} ${path}/${saved_in}_downsampled_${downsample_read}/chr${chrom}_${resolution}.txt
done
echo "  ...Downsampled contact matrix are generated: ./${saved_in}_downsampled_${downsample_read}/"
echo ""
