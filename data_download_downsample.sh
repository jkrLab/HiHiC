#!/bin/bash
set -e  # Error handling: stop script on any error
set -u  # Treat unset variables as an error
seed=42

# examples
# bash data_download_downsample.sh https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551550/suppl/GSM1551550_HIC001_merged_nodups.txt.gz GSM1551550_HIC001 hg19 5000000 ./juicer_tools.jar KR 10000 data
# bash data_download_downsample.sh https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551620/suppl/GSM1551620_HIC071_merged_nodups.txt.gz GSM1551620_HIC071 hg19 5000000 ./juicer_tools.jar KR 10000 data_K562
# bash data_download_downsample.sh https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551643/suppl/GSM1551643_HIC098_merged_nodups.txt.gz GSM1551643_HIC098 mm9.chrom.sizes 5000000 ./juicer_tools.jar KR 10000 data_ch12-lx

# 필요한 매개변수 체크
if [ "$#" -ne 8 ]; then
  echo "Usage: $0 <download_url> <file_name> <ref_genome> <downsampled_read> <juicertools> <normalization> <resolution> <saved_in>"
  exit 1
fi

### you can modifiy here as string type with ""!!!
path=$(pwd)
download_url=$1
file_name=$2
ref_genome=$3
downsampled_read=$4
juicertools=$5
normalization=$6
resolution=$7
saved_in=$8

echo ""
echo "  ...current working directory is $(pwd)"
echo "      data download url : ${download_url}"
echo "      raw data file : ${file_name}.txt.gz"
echo "      refrence genome : ${ref_genome}"
echo "      downsampled read : ${downsampled_read}"
echo "      juicertools : ${juicertools}" # path to juicer_tool.jar of docker image
echo "      normalization method : ${normalization}"
echo "      binning resolution : ${resolution}"
echo "      intra chromosome data : ${saved_in}/ and ${saved_in}_downsampled_${downsampled_read}/"
echo ""

### read data download ###
echo "  ...For download,"
save_name="${file_name}.txt.gz"
wget "${download_url}" -O "${save_name}" 


### downsampling ###
# 텍스트 파일 경로 설정
file_path=${path}/${save_name}

# 저장할 샘플링 파일 경로 설정
downsample_name=${path}/${file_name}_ds_${downsampled_read}.txt.gz

# 파일의 총 라인 수 가져오기
total_lines=$(zcat "$file_path" | wc -l)
echo ""
echo "  ...The number of total reads is ${total_lines}."

# 중복 없이 무작위로 샘플링한 라인을 저장할 임시 파일 생성
temp_file=$(mktemp)

# 원본 파일에서 중복 없이 무작위로 샘플링한 라인을 추출하여 임시 파일에 저장
zcat "$file_path" | shuf -n "$sample_size" >> "$temp_file"
echo "  ...And ${downsampled_read} reads are randomly sampled."

# 3번째와 7번째 컬럼(크로모좀) 기준으로 오름차순 정렬
sort -k3,3d -k7,7 "$temp_file" | gzip > "${downsample_name}"
rm "$temp_file"  # 임시 파일 삭제
echo "  ...Randomly sampled reads are saved: ${downsample_name}"


### .hic format ###
# juicer pre 로 .hic 변환
echo ""
echo "  ...To hic,"
java -Xmx2g -jar ${juicertools} pre ${save_name} ./${file_name}.hic ${ref_genome} # 원본 데이터
echo "  ...${file_name}.hic is generated."
echo ""
java -Xmx2g -jar ${juicertools} pre ${downsample_name} ./${file_name}_ds_${downsampled_read}.hic ${ref_genome} # 다운 샘플링 데이터
echo "  ...${file_name}_ds_${downsampled_read}.hic is generated."


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
fi

# 결과를 저장할 폴더 생성​​
mkdir ${path}/${saved_in}
# 전체 크로모좀에 대하여 반복문으로 명령 실행
for ((chrom=1; chrom<end; chrom++))
do
    java -jar ${juicertools} dump observed ${normalization} ./${file_name}.hic ${chrom} ${chrom} BP ${resolution} ${path}/${saved_in}/chr${chrom}_${resolution}.txt
done
echo ""
echo "  ...Intra chromosome contact matrix are generated: ./${saved_in}/"

# 결과를 저장할 폴더 생성​​
mkdir ${path}/${saved_in}_downsampled_${downsampled_read}
# 전체 크로모좀에 대하여 반복문으로 명령 실행
for ((chrom=1; chrom<end; chrom++))
do
    java -jar ${juicertools} dump observed ${normalization} ./${file_name}_ds_${downsampled_read}.hic ${chrom} ${chrom} BP ${resolution} ${path}/${saved_in}_downsampled_${downsampled_read}/chr${chrom}_${resolution}.txt
done
echo "  ...Downsampled contact matrix are generated: ./${saved_in}_downsampled_${downsampled_read}/"
echo ""