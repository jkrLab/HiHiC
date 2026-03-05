#!/bin/bash
set -euo pipefail

seed=${seed:-13}             
path=$(pwd)

# -----------------------------
# RAM & CPU 감지 및 제한 설정
# -----------------------------
TOTAL_MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
MAX_MEM_KB=$((128*1024*1024))                  
USE_MEM_KB=$(( TOTAL_MEM_KB / 10 )) # 2
(( USE_MEM_KB > MAX_MEM_KB )) && USE_MEM_KB=$MAX_MEM_KB
BUFFER_SIZE="${USE_MEM_KB}K"

TOTAL_CORES=$(nproc || echo 1)
MAX_CORES=4 # 16
(( TOTAL_CORES > MAX_CORES )) && USE_CORES=$MAX_CORES || USE_CORES=$TOTAL_CORES

# -----------------------------
# 임시 디렉토리 생성
# -----------------------------
# TMPDIR_PARENT="${TMPDIR:-/tmp}"
TMPDIR_PARENT="${TMPDIR:-$(dirname "$path")}" # 압축해제시 용량이 매우 크므로, /tmp 지정 
WORKDIR="$(mktemp -d -p "$TMPDIR_PARENT" hic.$$.XXXXXX)" 
cleanup() { [[ -d "$WORKDIR" ]] && rm -rf "$WORKDIR"; echo "  ...Temporary files deleted."; }
trap cleanup EXIT

# -----------------------------
# 읽기 수 줄임표기 함수
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

# -----------------------------
# resolution 줄임표기
# -----------------------------
if   (( resolution >= 1000000 )); then resolution_abbr=$(awk -v res="$resolution" 'BEGIN{printf "%dMb", res/1000000}')
elif (( resolution >= 1000    )); then resolution_abbr=$(awk -v res="$resolution" 'BEGIN{printf "%dKb", res/1000}')
else resolution_abbr=$resolution; fi

# -----------------------------
# 사용자 출력
# -----------------------------
echo ""
echo "  ...Current working directory: ${path}"
echo "    Raw data file: ${file_path}"
echo "    Data prefix: ${prefix}"
echo "    Reference genome: ${ref_genome}"
echo "    Reads to use: $([[ "$reads" -eq -1 ]] && echo "all" || echo "$reads")"
echo "    Juicertools: ${juicertools}"
echo "    Normalization: ${normalization}"
echo "    Resolution: ${resolution_abbr}"
echo "    Random seed: ${seed:-13}"
echo "    Output directory: ${saved_in}/${prefix}"
echo ""

# -----------------------------
# pigz / compress
# -----------------------------
DECOMP=$(command -v pigz >/dev/null 2>&1 && echo "pigz -dc" || echo "gzip -cd")
COMPRESS_CMD=$(command -v pigz >/dev/null 2>&1 && echo "pigz" || echo "gzip")

# -----------------------------
# 출력 디렉토리 생성
# -----------------------------
path="${saved_in}/${prefix}"
mkdir -p "${path}/READ" "${path}/HIC" "${path}/MAT"

# -----------------------------
# gzip 풀기 한 번만 수행 (temp_reads)
# -----------------------------
temp_reads="$WORKDIR/raw_reads.txt"
echo "  ...Decompressing input file (once)"
$DECOMP "$file_path" > "$temp_reads"

lines=$(wc -l < "$temp_reads")
reads_abbr=$(convert_reads "$reads")
lines_abbr=$(convert_reads "$lines")

# -----------------------------
# 다운샘플 및 저장
# -----------------------------
if [[ "$reads" -eq -1 || "$reads" -ge "$lines" ]]; then
    echo "  ...Using all reads, skipping sampling"
    downsample_name="${path}/READ/${prefix}__${lines_abbr}.txt.gz"
    cp "${file_path}" "$downsample_name"
    sort --buffer-size="${BUFFER_SIZE}" --parallel="${USE_CORES}" -k3,3V -k7,7V -k4,4n -k8,8n -T "$WORKDIR" "$temp_reads" | $COMPRESS_CMD > "$downsample_name"
else
    echo "  ...Sampling ${reads}/${lines} reads (seed=${seed})"
    downsample_name="${path}/READ/${prefix}__${lines_abbr}.txt.gz"
    temp_file="$WORKDIR/ds.sorted"
    awk -v k="$reads" -v seed="$seed" 'BEGIN { srand(seed) }
        NR <= k { res[NR]=$0; next }
        { j=int(rand()*NR)+1; if(j<=k) res[j]=$0 }
        END { for(i=1;i<=k;i++) if(i in res) print res[i] }' "$temp_reads" > "$temp_file"
    awk 'BEGIN{OFS="\t"}
      function ord(c,   t){
        t=c; sub(/^chr/,"",t);
        if (t ~ /^[0-9]+$/) return t+0;
        if (t=="X") return 1000;
        if (t=="Y") return 1001;
        if (t=="M"||t=="MT") return 1002;
        return 10000;
      }
      {
        chr1=$3; pos1=($4+0); chr2=$7; pos2=($8+0);
        o1=ord(chr1); o2=ord(chr2);
        if (o1>o2 || (o1==o2 && pos1>pos2)) {
          tmp3=$3; tmp4=$4; tmp5=$5; tmp6=$6;
          $3=$7; $4=$8; $5=$9; $6=$10;
          $7=tmp3; $8=tmp4; $9=tmp5; $10=tmp6;
        }
        print $0;
      }' "$temp_file" | \
      sort --buffer-size="${BUFFER_SIZE}" --parallel="${USE_CORES}" -k3,3V -k7,7V -k4,4n -k8,8n -T "$WORKDIR" "$temp_reads" | $COMPRESS_CMD > "$downsample_name"
    fi
echo "  ...Processed reads saved to: ${downsample_name}"

# -----------------------------
# .hic 생성
# -----------------------------
JAVA_MEM_MB=$(( TOTAL_MEM_KB*85/100/1024 ))
(( JAVA_MEM_MB < 2048 )) && JAVA_MEM_MB=2048
hic_file="${path}/HIC/${prefix}__${lines_abbr}.hic"
echo "  ...Generating .hic file (Xmx=${JAVA_MEM_MB}m)"
java -Xmx${JAVA_MEM_MB}m -jar "${juicertools}" pre "$downsample_name" "$hic_file" "$ref_genome"
echo "  ...hic file created: $hic_file"

# -----------------------------
# 크로모좀 리스트
# -----------------------------
chrom_list=()
case "$ref_genome" in
  *hg*|*GRCh*) for c in {1..22} X Y; do chrom_list+=("$c"); done;;
  *mm*|*GRCm*) for c in {1..19} X Y; do chrom_list+=("$c"); done;;
  *) 
    echo "Unknown reference genome: $ref_genome"
    read -p "Start chromosome (number): " start
    read -p "End chromosome (number): " end
    for ((c=start; c<=end; c++)); do chrom_list+=("$c"); done;;
esac

# -----------------------------
# intra-chromosome matrix 생성
# -----------------------------
mat_dir="${path}/MAT/${prefix}__${lines_abbr}_${resolution_abbr}_${normalization}"
mkdir -p "$mat_dir"
echo "  ...Dumping observed matrices (${normalization}, ${resolution_abbr})"
for chrom in "${chrom_list[@]}"; do
    java -Xmx${JAVA_MEM_MB}m -jar "${juicertools}" dump observed "${normalization}" "$hic_file" "$chrom" "$chrom" BP "$resolution" "${mat_dir}/chr${chrom}.txt" || true 
done
echo "  ...Contact matrices saved in: ${mat_dir}"

echo "All processes completed successfully."


bash data_generate_for_prediction.sh -i ${mat_dir} -b ${resolution} -m DFHiC -g ${ref_genome} -o ./data_original -s 250
