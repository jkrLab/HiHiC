#!/bin/bash
set -euo pipefail
seed=13
path=$(pwd)

# 임시 파일 정리를 위한 함수 정의
cleanup() {
  if [[ -f "${temp_file:-}" ]]; then
    rm -f "$temp_file"
    echo "  ...Temporary file deleted."
  fi
}
trap cleanup EXIT  # 스크립트 종료 시 cleanup 호출

while getopts ":i:p:g:r:j:n:b:o:" flag; 
do
    case $flag in
        i) download_url=$OPTARG;;
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
if [ -z "${download_url}" ] || [ -z "${prefix}" ] || [ -z "${ref_genome}" ] || [ -z "${reads}" ] || [ -z "${juicertools}" ] || [ -z "${normalization}" ]|| [ -z "${saved_in}" ]; then
    echo "Usage: $0 -i <input_data_URL> -p <prefix> -g <ref_genome> -r <downsample_reads> -j <juicertools> -n <normalization> -b <resolution> -o <output_directory>" >&2
    exit 1
fi

if (( reads >= 1000000 )); then
    sample_abbr=$(awk -v res="$reads" 'BEGIN { printf "%.1fM", res/1000000 }')
elif (( reads >= 1000 )); then
    sample_abbr=$(awk -v res="$reads" 'BEGIN { printf "%.1fK", res/1000 }')
else
    sample_abbr=$reads
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
echo "      Download url: ${download_url}"
echo "      Data prefix: ${prefix}"
echo "      Reference genome: ${ref_genome}"
echo "      Downsample read: ${reads}"
echo "      Juicertools: ${juicertools}"
echo "      Normalization method: ${normalization}"
echo "      Binning resolution: ${resolution_abbr}"
echo "      Output data: ${saved_in}/${prefix}/MAT."
echo ""

path=${saved_in}/${prefix}

### 다운로드 ###
echo "  ...Downloading data..."

mkdir -p "${path}/READ/"
save_name="${path}/READ/${prefix}.txt.gz"
if ! wget "${download_url}" -O "${save_name}"; then
  echo "Error: Failed to download file from ${download_url}"
  exit 1
fi

lines=$(zcat "${save_name}" | wc -l)
if (( lines >= 1000000 )); then
    reads_abbr=$(awk -v res="$lines" 'BEGIN { printf "%.1fM", res/1000000 }')
elif (( lines >= 1000 )); then
    reads_abbr=$(awk -v res="$lines" 'BEGIN { printf "%.1fK", res/1000 }')
else
    reads_abbr=$lines
fi

save_name="${path}/READ/${prefix}__${reads_abbr}.txt.gz"
mv "${path}/READ/${prefix}.txt.gz" ${save_name}

echo ""
echo "  ...Total reads in the file: ${reads_abbr}."

if [[ "$reads" -eq -1 ]]; then
  echo "  All reads (${lines}) will be sampled and made into contact map."
  reads=${lines}
  downsample_name=""
elif [ "${reads}" -lt "${lines}" ]; then
  downsample_name="${path}/READ/${prefix}__${sample_abbr}.txt.gz"
  temp_file=$(mktemp)
  # zcat "${save_name}" | shuf --random-seed=${seed} -n "${reads}" > "${temp_file}"
  zcat "${save_name}" | shuf -n "${reads}" > "${temp_file}"
  sort -k3,3d -k7,7 "${temp_file}" | gzip > "${downsample_name}"
  rm "${temp_file}"
  echo "  ...Downsampled reads saved: ${downsample_name}"
else
  echo "  ...Total reads (${lines}) are less than or equal to requested downsampled reads (${reads}). Skipping downsampling."
  downsample_name=""
fi

### .hic 파일 생성 ###
echo ""
echo "  ...Generating .hic files..."
mkdir -p "${path}/HIC"
java -Xmx2g -jar "${juicertools}" pre "${save_name}" "${path}/HIC/${prefix}__${reads_abbr}.hic" "${ref_genome}"
echo "  ...hic file created: ${path}/HIC/${prefix}__${reads_abbr}.hic"

if [ -n "$downsample_name" ]; then
  java -Xmx2g -jar "${juicertools}" pre "${downsample_name}" "${path}/HIC/${prefix}__${sample_abbr}.hic" "${ref_genome}"
  echo "  ...Downsampled .hic file created: ${path}/HIC/${prefix}__${sample_abbr}.hic"
fi

### 레퍼런스 지놈 크로모좀 범위 지정 ###
case "$ref_genome" in
  hg*) start=1; end=23 ;;
  mm*) start=1; end=20 ;;
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
echo "  ...Contact matrices saved in: ${path}/MAT/${prefix}__${reads_abbr}_${resolution_abbr}_${normalization}/"

if [ -n "$downsample_name" ]; then
  mkdir -p "${path}/MAT/${prefix}__${sample_abbr}_${resolution_abbr}_${normalization}"
  for ((chrom=start; chrom<=end; chrom++)); do
    java -jar "${juicertools}" dump observed "${normalization}" "${path}/HIC/${prefix}__${sample_abbr}.hic" "${chrom}" "${chrom}" BP "${resolution}" \
      "${path}/MAT/${prefix}__${sample_abbr}_${resolution_abbr}_${normalization}/chr${chrom}.txt"
  done
  echo "  ...Downsampled contact matrices saved in: ${path}/MAT/${prefix}__${sample_abbr}_${resolution_abbr}_${normalization}/"
fi
echo "All processes completed successfully."
echo ""
