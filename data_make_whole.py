import os, random, argparse
from multiprocessing import Pool
import numpy as np

path = os.getcwd()
random.seed(100)

# 인자 받기
parser = argparse.ArgumentParser(description='Read Hi-C contact map and Divide submatrix for train and predict', add_help=True)
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument('-i', '--input_data', dest='input_data', type=str, nargs='+', required=True, help='Input of model prediction')  # 여러 파일 및 디렉토리 받기
required.add_argument('-o', '--output_dir', dest='output_dir', type=str, required=True, help='Parental directory to save')
optional = parser.add_argument_group('optional arguments')
optional.add_argument('-w', '--workers', dest='workers', type=int, required=False, default=os.cpu_count(), help='Number of worker processes for reconstruction')
args = parser.parse_args()

def make_whole_40(predicted, save_filename):
    predicted = np.load(predicted, allow_pickle=True)
    mats = {}
    unique_chrom = np.unique(predicted['inds'][:, 0])
    for chrom in sorted(unique_chrom):
        chrom_indices = predicted['inds'][:, 0] == chrom
        mat_dim = int(predicted['inds'][chrom_indices][0, 1] + 1)
        inds = predicted['inds'][chrom_indices][:, -2:] + [6, 6]
        submats = np.squeeze(predicted['data'])[chrom_indices, 6:34, 6:34]
        mat = np.zeros((mat_dim, mat_dim), dtype=np.float32)
        for ind, submat in zip(inds, submats):
            mat[ind[0]:ind[0]+28, ind[1]:ind[1]+28] = submat
        mats[str(chrom)] = np.triu(mat) + np.triu(mat, k=1).T
    np.savez_compressed(save_filename, **mats)

def make_whole_28(predicted, save_filename):
    predicted = np.load(predicted, allow_pickle=True)
    mats = {}
    unique_chrom = np.unique(predicted['inds'][:, 0])
    for chrom in sorted(unique_chrom):
        chrom_indices = predicted['inds'][:, 0] == chrom
        mat_dim = int(predicted['inds'][chrom_indices][0, 1] + 1)
        inds = predicted['inds'][chrom_indices][:, -2:]
        submats = np.squeeze(predicted['data'])[chrom_indices, :, :]
        mat = np.zeros((mat_dim, mat_dim), dtype=np.float32)
        for ind, submat in zip(inds, submats):
            mat[ind[0]:ind[0]+28, ind[1]:ind[1]+28] = submat
        mats[str(chrom)] = np.triu(mat) + np.triu(mat, k=1).T
    np.savez_compressed(save_filename, **mats)


def _process_one_file(task):
    input_file, base_out = task
    try:
        prefix = os.path.splitext(os.path.basename(input_file))[0].split("__")[0]
        output_dir = os.path.join(base_out, prefix)
        os.makedirs(output_dir, exist_ok=True)
        save_filename = os.path.join(output_dir, os.path.basename(input_file))
        data = np.load(input_file, allow_pickle=True)
        data_shape = int(np.squeeze(data['data']).shape[-1])
        if data_shape == 40:
            make_whole_40(input_file, save_filename)
        elif data_shape == 28:
            make_whole_28(input_file, save_filename)
        else:
            return f"{input_file} isn't 40*40 or 28*28 sub matrix."
        return f"{save_filename} was done."
    except Exception as e:
        return f"Error processing {input_file}: {e}"

# Build task list
npz_files = []
for data_path in args.input_data:
    if os.path.isdir(data_path):
        for filename in os.listdir(data_path):
            candidate = os.path.join(data_path, filename)
            if candidate.endswith('.npz'):
                npz_files.append(candidate)
    else:
        if data_path.endswith('.npz'):
            npz_files.append(data_path)

if len(npz_files) == 0:
    raise SystemExit("No .npz files found in given inputs.")

workers = max(1, int(args.workers) if args.workers else os.cpu_count())
tasks = [(f, args.output_dir) for f in npz_files]
with Pool(processes=min(workers, len(npz_files))) as pool:
    for msg in pool.imap_unordered(_process_one_file, tasks):
        print(msg)