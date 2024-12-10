import os, random, argparse
import numpy as np

path = os.getcwd()
random.seed(100)

# 인자 받기
parser = argparse.ArgumentParser(description='Read Hi-C contact map and Divide submatrix for train and predict', add_help=True)
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument('-i', '--input_data', dest='input_data', type=str, nargs='+', required=True, help='Input of model prediction')  # 여러 파일 및 디렉토리 받기
required.add_argument('-o', '--output_dir', dest='output_dir', type=str, required=True, help='Parental directory to save')

args = parser.parse_args()
input_data = args.input_data  # 리스트 형태로 받음
output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(args.input_data))[0].split("__")[0])
os.makedirs(output_dir, exist_ok=True)


def make_whole_40(predicted, save_filename):
    predicted = np.load(predicted, allow_pickle=True)
    mats = {}
    unique_chrom = np.unique(predicted['inds'][:, 0])
    for chrom in sorted(unique_chrom):
        chrom_indices = predicted['inds'][:, 0] == chrom
        mat_dim = predicted['inds'][chrom_indices][0, 1] + 1
        inds = predicted['inds'][chrom_indices][:, -2:] + [6, 6]
        submats = np.squeeze(predicted['data'])[chrom_indices, 6:34, 6:34]
        mat = np.zeros((mat_dim, mat_dim))
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
        mat_dim = predicted['inds'][chrom_indices][0, 1] + 1
        inds = predicted['inds'][chrom_indices][:, -2:]
        submats = np.squeeze(predicted['data'])[chrom_indices, :, :]
        mat = np.zeros((mat_dim, mat_dim))
        for ind, submat in zip(inds, submats):
            mat[ind[0]:ind[0]+28, ind[1]:ind[1]+28] = submat
        mats[str(chrom)] = np.triu(mat) + np.triu(mat, k=1).T
    np.savez_compressed(save_filename, **mats)


# for data_file in input_data:
if os.path.isdir(data_file):  # 디렉토리인 경우
    for filename in os.listdir(data_file):
        input_file_path = os.path.join(data_file, filename)
        if input_file_path.endswith('.npz'):  # 원하는 파일 형식 필터링
            save_filename = os.path.join(output_dir, filename)
            data_shape = np.squeeze(np.load(input_file_path, allow_pickle=True)['data']).shape[-1]
            if data_shape == 40:
                make_whole_40(input_file_path, save_filename)
                print(f'{save_filename} was done.')
            elif data_shape == 28:
                make_whole_28(input_file_path, save_filename)
                print(f'{save_filename} was done.')
            else:
                print(f"The output of iEnhance in {input_file_path} doesn't need a chromosome matrix.")
else:  # 파일인 경우
    if data_file.endswith('.npz'):
        save_filename = os.path.join(output_dir, filename)
        data_shape = np.squeeze(np.load(data_file, allow_pickle=True)['data']).shape[-1]
        if data_shape == 40:
            make_whole_40(data_file, save_filename)
            print(f'{save_filename} was done.')
        elif data_shape == 28:
            make_whole_28(data_file, save_filename)
            print(f'{save_filename} was done.')
        else:
            print(f"The output of iEnhance in {data_file} doesn't need a chromosome matrix.")