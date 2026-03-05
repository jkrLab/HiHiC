#!/usr/bin/env python3
# npz_to_tsv.py (중복 검사 기능 추가 버전)
import os, sys, argparse, numpy as np, pandas as pd, zipfile
from tqdm import tqdm
from collections import Counter

def chromosome_sort_key(filename):
    chrom = filename.replace('.npy', '')
    if chrom.startswith('chr'): chrom = chrom.replace('chr', '')
    if chrom.isdigit(): return int(chrom)
    elif chrom.upper() == 'X': return 100
    elif chrom.upper() == 'Y': return 101
    else: return 200

def map_chrom(chrom, genome="hg19"):
    if chrom.startswith('chr'):
        return chrom # 이미 'chr1', 'chrX' 형태이므로 그대로 반환
    
    if genome == "mm9":
        if chrom == "20": return "chrX"
        if chrom == "21": return "chrY"
    elif genome == "hg19":
        if chrom == "23" or chrom == "100": return "chrX"
        if chrom == "24" or chrom == "101": return "chrY"
    return "chr" + chrom

def load_chromsizes(ref_file):
    chromsizes = {}
    with open(ref_file) as f:
        for line in f: chrom, size = line.strip().split(); chromsizes[chrom] = int(size)
    return chromsizes

def convert_npz_to_tsv(npz_path, tsv_path, chromsizes_path, binsize, genome_name):
    if not os.path.exists(npz_path):
        print(f"❌ Error: Input NPZ file not found at {npz_path}"); sys.exit(1)
    chromsizes = load_chromsizes(chromsizes_path)
    bin_offset = 0
    try:
        with open(tsv_path, 'w') as f_out:
            f_out.write("bin1_id\tbin2_id\tcount\n")
            with zipfile.ZipFile(npz_path, 'r') as zf:
                npy_files = [f for f in zf.namelist() if f.endswith('.npy')]
                chrom_files = sorted(npy_files, key=chromosome_sort_key)
                print(chrom_files)
                for npy_file in tqdm(chrom_files, desc=f"Converting {os.path.basename(npz_path)}"):
                    chrom = npy_file.replace('.npy', '')
                    with zf.open(npy_file) as f:
                        mat = np.load(f).astype(np.float32) * 250
                    # mat += mat.T - np.diag(np.diag(mat)) # 대칭 행렬로 변환
                    chrom_ref = map_chrom(chrom, genome=genome_name)
                    if chrom_ref not in chromsizes: continue
                    n_bins_ref = (chromsizes[chrom_ref] + binsize - 1) // binsize
                    print(chrom_ref, " size :", n_bins_ref)
                    if mat.shape[0] < n_bins_ref: # 매트릭스 작은 경우 패딩
                        pad = n_bins_ref - mat.shape[0]; mat = np.pad(mat, ((0, pad), (0, pad)))
                    rows, cols = np.nonzero(mat); vals = mat[rows, cols]
                    rows_global, cols_global = rows + bin_offset, cols + bin_offset
                    mask = rows_global <= cols_global
                    bin1s, bin2s, counts = rows_global[mask].astype(np.int32), cols_global[mask].astype(np.int32), vals[mask].astype(np.float32)
                    if len(bin1s) > 0: print(" start bin :", bin1s[0], bin2s[0],)

                    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
                    # 중복 검사 로직 추가
                    # 생성된 (bin1, bin2) 쌍의 개수와, 중복을 제거한 유니크한 쌍의 개수를 비교
                    pixel_pairs = set(zip(bin1s, bin2s))
                    if len(pixel_pairs) < len(bin1s):
                        print(f"\n❌ FATAL ERROR: Intra-chromosomal duplicate pixels found in chromosome '{chrom}' of file {os.path.basename(npz_path)}!")
                        
                        # 어떤 값이 중복되었는지 샘플 출력
                        duplicates = [item for item, count in Counter(zip(bin1s, bin2s)).items() if count > 1]
                        print(f"Example duplicates found: {duplicates[:5]}")
                        
                        sys.exit(1) # 중복 발견 시 즉시 종료
                    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

                    for i in range(len(bin1s)):
                        f_out.write(f"{bin1s[i]}\t{bin2s[i]}\t{counts[i]}\n")
                    bin_offset += n_bins_ref
                    del mat, rows, cols, vals, rows_global, cols_global, mask, bin1s, bin2s, counts
    except Exception as e:
        print(f"❌ An error occurred while processing {npz_path}: {e}"); sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Convert NPZ to an intermediate TSV format.")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-g", "--chromsizes", required=True)
    parser.add_argument("-b", "--binsize", type=int, default=10000)
    parser.add_argument("--genome", type=str, default="hg19")
    args = parser.parse_args()
    convert_npz_to_tsv(args.input, args.output, args.chromsizes, args.binsize, args.genome)

if __name__ == "__main__":
    main()