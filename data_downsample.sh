#!/bin/bash

# # 임시 파일 정리를 위한 함수 정의
# cleanup() {
#   if [[ -f "$temp_file" ]]; then
#     rm -f "$temp_file"
#     echo "  ...Temporary file deleted."
#   fi
# }
# trap cleanup EXIT  # 스크립트 종료 시 cleanup 호출

# # examples
# # bash data_downsample.sh GSM1551550_HIC001 hg19 5000000 ./juicer_tools.jar KR 10000 data_GM12878
# # bash data_downsample.sh GSM1551620_HIC071 hg19 5000000 ./juicer_tools.jar KR 10000 data_K562
# # bash data_downsample.sh GSM1551643_HIC098 mm9.chrom.sizes 5000000 ./juicer_tools.jar KR 10000 data_ch12-lx

# # 필요한 매개변수 체크
# if [ "$#" -ne 7 ]; then
#   echo "Usage: $0 <file_prefix> <ref_genome> <downsample_read> <juicertools> <normalization> <resolution> <saved_in>"
#   exit 1
# fi

# ### you can modify here as string type with ""!!!
# path=$(pwd)
# file_name=$1
# ref_genome=$2
# downsample_read=$3
# juicertools=$4
# normalization=$5
# resolution=$6
# saved_in=$7

# echo ""
# echo "  ...current working directory is $(pwd)"
# echo "      raw data file : ${file_name}"
# echo "      refrence genome : ${ref_genome}"
# echo "      downsampling read : ${downsample_read}"
# echo "      juicertools : ${juicertools}" # path to juicer_tool.jar of docker image
# echo "      normalization method : ${normalization}"
# echo "      binning size : ${resolution} bp"
# echo "      intra chromosome data : ${saved_in}/ and ${saved_in}_downsampled_${downsample_read}/"
# echo ""

# save_name="${file_name}.txt.gz"
# file_path=${path}/${save_name}

# # 저장할 샘플링 파일 경로 설정
# downsample_name=${path}/${file_name}_ds_${downsample_read}.txt.gz

# # 파일의 총 라인 수 가져오기
# total_lines=$(zcat "$file_path" | wc -l)
# echo ""
# echo "  ...The number of total reads is ${total_lines}."

# # downsampled_read 확인 및 조건 처리
# if [ "$total_lines" -le "$downsampled_read" ]; then
#   echo "  ...Total reads (${total_lines}) are less than or equal to the requested downsampled reads (${downsample_read}). Skipping downsampling."
#   downsample_name="" # 이후에 관련 작업 건너뛰기 위해 빈 값 설정
# else
#   # 저장할 샘플링 파일 경로 설정
#   downsample_name=${path}/${file_name}_ds_${downsampled_read}.txt.gz

#   # 중복 없이 무작위로 샘플링한 라인을 저장할 임시 파일 생성
#   temp_file=$(mktemp)

#   # 원본 파일에서 중복 없이 무작위로 샘플링한 라인을 추출하여 임시 파일에 저장
#   zcat "$file_path" | shuf -n "$downsampled_read" >> "$temp_file"
#   echo "  ...And ${downsampled_read} reads are randomly sampled."

#   # 3번째와 7번째 컬럼(크로모좀) 기준으로 오름차순 정렬
#   sort -k3,3d -k7,7 "$temp_file" | gzip > "${downsample_name}"
#   rm "$temp_file"  # 임시 파일 삭제
#   echo "  ...Randomly sampled reads are saved: ${downsample_name}"
# fi


# ### .hic format ###
# # downsample_name이 존재하는 경우에만 .hic 파일 생성
# if [ -n "$downsample_name" ]; then
# echo ""
# echo "  ...To hic,"
#   java -Xmx2g -jar ${juicertools} pre ${downsample_name} ./${file_name}_ds_${downsampled_read}.hic ${ref_genome}
#   echo "  ...${file_name}_ds_${downsampled_read}.hic is generated."
# fi

# ### .txt format for each chromosome (intra) ###
# # ref_genome에 따라 크로모좀 범위 설정
# if [[ "$ref_genome" == hg* ]]; then
#   start=1
#   end=23
#   echo "Reference genome: ${ref_genome}: chr${start} - chr${end}"
# elif [[ "$ref_genome" == mm* ]]; then
#   start=1
#   end=20
#   echo "Reference genome: ${ref_genome}: chr${start} - chr${end}"
# else
#   echo "Unknown reference genome"
#   exit 1
# fi


# # 다운샘플링 데이터에 대해서는 downsample_name이 존재하는 경우에만 실행
# if [ -n "$downsample_name" ]; then
#   mkdir -p ${path}/${saved_in}_downsampled_${downsampled_read}
#   for ((chrom=1; chrom<end; chrom++))
#   do
#       java -jar ${juicertools} dump observed ${normalization} ./${file_name}_ds_${downsampled_read}.hic ${chrom} ${chrom} BP ${resolution} ${path}/${saved_in}_downsampled_${downsampled_read}/chr${chrom}_${resolution}.txt
#   done
#   echo "  ...Downsampled contact matrix are generated: ./${saved_in}_downsampled_${downsampled_read}/"
# fi
# echo ""


set -euo pipefail
seed=42

# 임시 파일 정리를 위한 함수
cleanup() {
  if [[ -n "${temp_file:-}" && -f "$temp_file" ]]; then
    rm -f "$temp_file"
    echo "  ...Temporary file deleted."
  fi
}
trap cleanup EXIT

# 필요한 매개변수 확인
if [ "$#" -ne 7 ]; then
  echo "Usage: $0 <file_prefix> <ref_genome> <downsample_read> <juicertools> <normalization> <resolution> <saved_in>"
  exit 1
fi

path=$(pwd)
file_name=$1
ref_genome=$2
downsampled_read=$3
juicertools=$4
normalization=$5
resolution=$6
saved_in=$7

# 확장자 확인 및 추가
if [[ ! "$file_name" =~ \.txt\.gz$ ]]; then
  file_name="${file_name}.txt.gz"
fi
file_path="${path}/${file_name}"

# 파일 존재 여부 확인
if [[ ! -f "$file_path" ]]; then
  echo "Error: File '$file_path' does not exist."
  exit 1
fi

echo "Processing with:"
echo "  Raw data file: $file_name"
echo "  Reference genome: $ref_genome"
echo "  Downsample reads: $downsampled_read"
echo "  Juicer tools: $juicertools"
echo "  Normalization: $normalization"
echo "  Resolution: ${resolution}bp"

downsample_name="${path}/${file_name}_ds_${downsampled_read}.txt.gz"
total_lines=$(zcat "$file_path" | wc -l)

if [[ "$total_lines" -le "$downsampled_read" ]]; then
  echo "  Skipping downsampling: total reads ($total_lines) <= requested reads ($downsampled_read)"
  downsample_name=""
else
  temp_file=$(mktemp)
  zcat "$file_path" | shuf -n "$downsampled_read" > "$temp_file"
  gzip < "$temp_file" > "$downsample_name"
  echo "  Downsampled reads saved to: $downsample_name"
fi

# .hic 파일 생성
if [[ -n "$downsample_name" ]]; then
  java -Xmx2g -jar "$juicertools" pre "$downsample_name" "./${file_name}_ds_${downsampled_read}.hic" "$ref_genome"
fi

# 크로모좀 범위 확인
case "$ref_genome" in
  hg*) start=1; end=23 ;;
  mm*) start=1; end=20 ;;
  *) echo "Unknown reference genome: $ref_genome"; exit 1 ;;
esac

# 다운샘플링 데이터로 크로모좀 매트릭스 생성
if [[ -n "$downsample_name" ]]; then
  output_dir="${path}/${saved_in}_downsampled_${downsampled_read}"
  mkdir -p "$output_dir"
  for ((chrom=start; chrom<end; chrom++)); do
    java -jar "$juicertools" dump observed "$normalization" "./${file_name}_ds_${downsampled_read}.hic" \
      "$chrom" "$chrom" BP "$resolution" "${output_dir}/chr${chrom}_${resolution}.txt"
  done
  echo "  Downsampled contact matrices saved to: $output_dir"
fi
