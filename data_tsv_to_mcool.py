#!/usr/bin/env python3
# tsv_to_mcool.py
import os, argparse, pandas as pd, cooler, h5py, subprocess

def make_bins_dict(chromsizes, binsize):
    chrom_list, start_list, end_list = [], [], []
    for chrom, size in chromsizes.items():
        for start in range(0, size, binsize):
            chrom_list.append(chrom)
            start_list.append(start)
            end_list.append(min(start + binsize, size))
    return pd.DataFrame({"chrom": chrom_list, "start": start_list, "end": end_list})

def load_chromsizes(ref_file):
    chromsizes = {}
    with open(ref_file) as f:
        for line in f:
            chrom, size = line.strip().split()
            chromsizes[chrom] = int(size)
    return chromsizes

def create_cooler_from_tsv(tsv_path, out_dir, file_name_base, chromsizes_path, binsize, resolutions, make_cool, make_mcool):
    output_cool = os.path.join(out_dir, file_name_base + ".cool")
    output_mcool = os.path.join(out_dir, file_name_base + ".mcool")
    chromsizes = load_chromsizes(chromsizes_path)
    bins = make_bins_dict(chromsizes, binsize)
    chunk_size = 10_000_000
    pixels_iterator = pd.read_csv(tsv_path, sep='\t', chunksize=chunk_size, dtype={'bin1_id': int, 'bin2_id': int, 'count': float})
    
    # [참고] ordered=True로 변경하면 성능이 향상될 수 있습니다. 
    # (이전 단계에서 sort | awk를 사용했다면)
    cooler.create_cooler(output_cool, bins=bins, pixels=pixels_iterator, ordered=False) 
    if make_cool:
         with h5py.File(output_cool, "r+") as f:
            f.attrs["normalization"] = "KR"
    if make_mcool:
        cmd = ["cooler", "zoomify", output_cool, "-o", output_mcool, "-r", ",".join(map(str, resolutions))]
        subprocess.run(cmd, check=True)
        if not make_cool: os.remove(output_cool)
    print(f"✅ Finished creating cooler files for {file_name_base}")

def main():
    parser = argparse.ArgumentParser(description="Create .cool/.mcool from an intermediate TSV file.")
    parser.add_argument("-i", "--input", required=True, help="Input .tsv file")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-n", "--name", required=True, help="Base name for the output file (without extension)")
    parser.add_argument("-g", "--chromsizes", required=True, help="Reference chrom.sizes file")
    parser.add_argument("-b", "--binsize", type=int, default=10000)
    parser.add_argument("-r", "--resolutions", nargs="+", type=int, default=[10000, 20000, 50000, 100000])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cool-only", action="store_true")
    group.add_argument("--mcool-only", action="store_true")
    args = parser.parse_args()
    make_cool, make_mcool = not args.mcool_only, not args.cool_only
    create_cooler_from_tsv(args.input, args.output, args.name, args.chromsizes, args.binsize, args.resolutions, make_cool, make_mcool)

if __name__ == "__main__":
    main()