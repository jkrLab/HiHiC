#!/bin/bash
set -euo pipefail

seed=${seed:-13}             # 셔플 시드 (환경변수/기본 13)
path=$(pwd)

# -----------------------------
# RAM & CPU 감지 및 제한 설정
# -----------------------------
TOTAL_MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
MAX_MEM_KB=$((64*1024*1024))                   # 64GB 상한
USE_MEM_KB=$(( TOTAL_MEM_KB / 2 ))
(( USE_MEM_KB > MAX_MEM_KB )) && USE_MEM_KB=$MAX_MEM_KB
BUFFER_SIZE="${USE_MEM_KB}K"

TOTAL_CORES=$(nproc || echo 1)
MAX_CORES=16
(( TOTAL_CORES > MAX_CORES )) && USE_CORES=$MAX_CORES || USE_CORES=$TOTAL_CORES

# -----------------------------
# 임시 디렉토리/파일 정리
# -----------------------------
TMPDIR_PARENT="${TMPDIR:-/tmp}"
WORKDIR="$(mktemp -d -p "$TMPDIR_PARENT" hic.$$.XXXXXX)"
temp_file="$WORKDIR/ds.sorted"
cleanup() {
  if [[ -d "$WORKDIR" ]]; then
    rm -rf "$WORKDIR"
    echo "  ...Temporary files deleted."
  fi
}
trap cleanup EXIT

# -----------------------------
# 읽기 수 줄임표기
# -----------------------------
convert_reads() {
  local r=$1
  if   (( r >= 1000000 )); then awk -v x="$r" 'BEGIN{printf "%.1fM", x/1000000}'
  elif (( r >= 1000    )); then awk -v x="$r" 'BEGIN{printf "%.1fK", x/1000}'
  else echo "$r"
  fi
}

# -----------------------------
# 옵션 파싱
# -----------------------------
while getopts ":i:p:g:r:j:n:b:o:s:" flag; do
  case $flag in
    i) file_path=$OPTARG;;
    p) prefix=$OPTARG;;
    g) ref_genome=$OPTARG;;
    r) reads=$OPTARG;;
    j) juicertools=$OPTARG;;
    n) normalization=$OPTARG;;
    b) resolution=$OPTARG;;
    o) saved_in=$OPTARG;;
    s) seed=$OPTARG;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1;;
    :)  echo "Option -$OPTARG requires an argument." >&2; exit 1;;
  esac
done

# 필수 인자 체크
if [ -z "${file_path:-}" ] || [ -z "${prefix:-}" ] || [ -z "${ref_genome:-}" ] || \
   [ -z "${reads:-}" ] || [ -z "${juicertools:-}" ] || [ -z "${normalization:-}" ] || \
   [ -z "${saved_in:-}" ]; then
  echo "Usage: $0 -i <input_data_path> -p <prefix> -g <ref_genome> -r <downsample_reads> -j <juicertools> -n <normalization> -b <resolution> -o <output_directory> [-s <random_seed>]" >&2
  exit 1
fi

[[ ! -f "$file_path" ]] && echo "Error: File '$file_path' does not exist." && exit 1

# resolution 줄임표기
if   (( resolution >= 1000000 )); then resolution_abbr=$(awk -v res="$resolution" 'BEGIN{printf "%dMb", res/1000000}')
elif (( resolution >= 1000    )); then resolution_abbr=$(awk -v res="$resolution" 'BEGIN{printf "%dKb", res/1000}')
else resolution_abbr=$resolution; fi

echo ""
echo "  ...Current working directory: ${path}"
echo "    Raw data file: ${file_path}"
echo "    Data prefix: ${prefix}"
echo "    Reference genome: ${ref_genome}"
echo "    Downsample reads: ${reads}"
echo "    Juicertools: ${juicertools}"
echo "    Normalization: ${normalization}"
echo "    Resolution: ${resolution_abbr}"
echo "    Random seed: ${seed:-13}"
echo "    Output data: ${saved_in}/${prefix}/MAT."
echo ""

# -----------------------------
# pigz 여부 => 멀티코어 / pv 여부 => 프로그레션바
# -----------------------------
if command -v pigz >/dev/null 2>&1; then
  DECOMP="pigz -dc"
else
  DECOMP="gzip -cd"
fi
lines=$($DECOMP "$file_path" | wc -l)

if command -v pigz >/dev/null 2>&1; then
    COMPRESS_CMD="pigz"
else
    COMPRESS_CMD="gzip"
fi

# pv가 없으면 cat으로 대체
if command -v pv >/dev/null 2>&1; then
    PV_CMD=(pv -l -s "$lines")
else
    PV_CMD=(cat)
fi

# 다운샘플 읽기 수 결정
if [[ "$reads" -eq -1 || "$reads" -ge "$lines" ]]; then
  echo "  All reads (${lines}) will be sampled and made into contact map."
  reads=${lines}
fi
reads_abbr=$(convert_reads "$reads")

# 출력 경로
path="${saved_in}/${prefix}"
mkdir -p "${path}/READ" "${path}/HIC"
downsample_name="${path}/READ/${prefix}__${reads_abbr}.txt.gz"



# -----------------------------
# pigz 멀티코어 / pv 프로그레션바
# -----------------------------
if [[ "$reads" -lt "$lines" ]]; then
    echo "  ...Sampling ${reads}/${lines} with reservoir sampling (seed=${seed})"
    $DECOMP "$file_path" | "${PV_CMD[@]}" |
    awk -v k="$reads" -v seed="$seed" '
    BEGIN { srand(seed) }
    NR <= k { res[NR] = $0; next }
    {  j = int(rand()*NR) + 1
      if (j <= k) res[j] = $0 }
    END { for (i = 1; i <= k; i++) if (i in res) print res[i] }' > "$temp_file"

    echo "  ...Sorting reads"
    "${PV_CMD[@]}" "$temp_file" | sort --buffer-size="${BUFFER_SIZE}" --parallel="${USE_CORES}" -T "$WORKDIR" -k3,3d -k7,7 | $COMPRESS_CMD > "$downsample_name"
else
    echo "  ...Sorting reads"
    $DECOMP "$file_path" | "${PV_CMD[@]}" | sort --buffer-size="${BUFFER_SIZE}" --parallel="${USE_CORES}" -T "$WORKDIR" -k3,3d -k7,7 | $COMPRESS_CMD > "${downsample_name}"
fi

echo "  Processed reads saved to: ${downsample_name}"


# -----------------------------
# .hic 파일 생성 (Java 메모리 동적)
# -----------------------------
# JAVA_MEM_MB=$(( $(( TOTAL_MEM_KB * 85 / 100 ))  / 1024 )) # 85%
JAVA_MEM_MB=$(( USE_MEM_KB / 1024 / 2 ))     # 전체 허용 절반
(( JAVA_MEM_MB < 2048 )) && JAVA_MEM_MB=2048 # 최소 2GB 권장
echo "  ...Generating .hic files (Xmx=${JAVA_MEM_MB}m)"
java -Xmx${JAVA_MEM_MB}m -jar "${juicertools}" pre "${downsample_name}" "${path}/HIC/${prefix}__${reads_abbr}.hic" "$ref_genome"
echo "  ...hic file created: ${path}/HIC/${prefix}__${reads_abbr}.hic"


# -----------------------------
# 크로모좀 리스트 (chrX/chrY 포함)
# -----------------------------
chrom_list=()
case "$ref_genome" in
  *hg*|*GRCh*)
    for c in {1..22} X Y; do chrom_list+=("$c"); done
    ;;
  *mm*|*GRCm*)
    for c in {1..19} X Y; do chrom_list+=("$c"); done
    ;;
  *)
    echo "Unknown reference genome: $ref_genome"
    read -p "Start chromosome (number): " start
    read -p "End chromosome (number): " end
    if ! [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ && "$start" -ge 1 && "$end" -ge "$start" ]]; then
      echo "Invalid chromosome range."; exit 1
    fi
    for ((c=start; c<=end; c++)); do chrom_list+=("$c"); done
    ;;
esac


# -----------------------------
# intra-chromosome contact matrices
# -----------------------------
mat_dir="${path}/MAT/${prefix}__${reads_abbr}_${resolution_abbr}_${normalization}"
mkdir -p "$mat_dir"
echo "  ...Dumping observed matrices (${normalization}, ${resolution_abbr})"
for chrom in "${chrom_list[@]}"; do
  java -Xmx${JAVA_MEM_MB}m -jar "${juicertools}" dump observed "${normalization}" \
    "${path}/HIC/${prefix}__${reads_abbr}.hic" "${chrom}" "${chrom}" BP "${resolution}" \
    "${mat_dir}/chr${chrom}.txt"
done
echo "  ...Downsampled contact matrices saved in: ${mat_dir}"
echo "All processes completed successfully."
