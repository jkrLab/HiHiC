#!/bin/bash
set -euo pipefail
seed=13
path=$(pwd)

# 임시 파일 정리를 위한 함수
cleanup() {
  if [[ -n "${temp_file:-}" && -f "$temp_file" ]]; then
    rm -f "$temp_file"
    echo "  ...Temporary file deleted."
  fi
}
trap cleanup EXIT

convert_reads() {
  local reads=$1
  local reads_abbr

  if (( reads >= 1000000 )); then
      reads_abbr=$(awk -v res="$reads" 'BEGIN { printf "%.1fM", res/1000000 }')
  elif (( reads >= 1000 )); then
      reads_abbr=$(awk -v res="$reads" 'BEGIN { printf "%.1fK", res/1000 }')
  else
      reads_abbr=$reads
  fi

  echo "$reads_abbr"
}

while getopts ":i:p:g:r:j:n:b:o:" flag; 
do
    case $flag in
        i) file_path=$OPTARG;;
        p) prefix=$OPTARG;;
        g) ref_genome=$OPTARG;;
        r) reads=$OPTARG;;
        j) juicertools=$OPTARG;;
        n) normalization=$OPTARG;;
        b) resolution=$OPTARG;;
        o) saved_in=$OPTARG;;
        \?)
        echo "Invalid option: -$OPTARG" >&2
        exit 1;;
        :)
        echo "Option -$OPTARG requires an argument." >&2
        exit 1;;
    esac
done

# 필수 인자 체크
if [ -z "${file_path}" ] || [ -z "${prefix}" ] || [ -z "${ref_genome}" ] || [ -z "${reads}" ] || [ -z "${juicertools}" ] || [ -z "${normalization}" ]|| [ -z "${saved_in}" ]; then
    echo "Usage: $0 -i <input_data_path> -p <prefix> -g <ref_genome> -r <downsample_reads> -j <juicertools> -n <normalization> -b <resolution> -o <output_directory>" >&2
    exit 1
fi

# 파일 존재 여부 확인
if [[ ! -f "${file_path}" ]]; then
  echo "Error: File '${file_path}' does not exist."
  exit 1
fi

if (( resolution >= 1000000 )); then
    resolution_abbr=$(awk -v res="$resolution" 'BEGIN { printf "%dMb", res/1000000 }')
elif (( resolution >= 1000 )); then
    resolution_abbr=$(awk -v res="$resolution" 'BEGIN { printf "%dKb", res/1000 }')
else
    resolution_abbr=$resolution
fi

echo ""
echo "  ...Current working directory: ${path}"
echo "    Raw data file: ${file_path}"
echo "    Data prefix: ${prefix}"
echo "    Reference genome: ${ref_genome}"
echo "    Downsample reads: ${reads}"
echo "    Juicertools: ${juicertools}"
echo "    Normalization method: ${normalization}"
echo "    Binning resolution: ${resolution_abbr}"
echo "    Output data: ${saved_in}/${prefix}/MAT."
echo ""

path=${saved_in}/${prefix}
lines=$(zcat "$file_path" | wc -l)

if [[ "$reads" -eq -1 ]]; then
  echo "  All reads (${lines}) will be sampled and made into contact map."
  reads=${lines}
  reads_abbr=$(convert_reads $reads)
  downsample_name=${file_path}
elif [ "$reads" -lt "$lines" ]; then
  reads_abbr=$(convert_reads $reads)
  mkdir -p "${path}/READ"
  downsample_name="${path}/READ/${prefix}__${reads_abbr}.txt.gz"
  temp_file=$(mktemp)
  # zcat "${save_name}" | shuf --random-seed=${seed} -n "${reads}" > "${temp_file}"
  zcat "${file_path}" | shuf -n "${reads}" > "${temp_file}"
  sort -k3,3d -k7,7 "${temp_file}" | gzip > "${downsample_name}"
  rm "${temp_file}"
  echo "  Downsampled reads saved to: ${downsample_name}"
else
  echo "  Skipping downsampling: total reads (${lines}), requested reads (${reads})"
  reads=${lines}
  reads_abbr=$(convert_reads $reads)
  downsample_name=${file_path}
fi

if [ -n "$downsample_name" ]; then
  # .hic 파일 생성
  echo ""
  echo "  ...Generating .hic files..."
  mkdir -p "${path}/HIC"
  java -Xmx4g -jar "${juicertools}" pre "${downsample_name}" "${path}/HIC/${prefix}__${reads_abbr}.hic" "$ref_genome"
  echo "  ...hic file created: ${path}/HIC/${prefix}__${reads_abbr}.hic"


  # 크로모좀 범위 확인
case "$ref_genome" in
  hg*) start=1; end=22 ;;
  mm*) start=1; end=19 ;;
  *)
    echo "Unknown reference genome: $ref_genome"
    echo "Please specify the chromosome range."
    read -p "Start chromosome: " start
    read -p "End chromosome: " end
  if ! [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ && "$start" -ge 1 && "$end" -ge "$start" ]]; then
    echo "Invalid chromosome range. Start and End must be positive integers, and Start <= End."
      exit 1
    fi
    ;;
esac

  ### intra-chromosome contact matrix 생성 ###
  mkdir -p "${path}/MAT/${prefix}__${reads_abbr}_${resolution_abbr}_${normalization}"
  for ((chrom=start; chrom<=end; chrom++)); do
    java -jar "${juicertools}" dump observed "${normalization}" "${path}/HIC/${prefix}__${reads_abbr}.hic" "${chrom}" "${chrom}" BP "${resolution}" \
      "${path}/MAT/${prefix}__${reads_abbr}_${resolution_abbr}_${normalization}/chr${chrom}.txt"
  done
  echo "  ...Downsampled contact matrices saved in: ${path}/MAT/${prefix}__${reads_abbr}_${resolution_abbr}_${normalization}/"
  echo "All processes completed successfully."
  echo ""
fi
