import os, random, argparse
import numpy as np

path = os.getcwd()
random.seed(100)

# 인자 받기
parser = argparse.ArgumentParser(description='Read Hi-C contact map and Divide submatrix for train and predict', add_help=True)
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-i', '--input_data', dest='input_data', type=str, required=True, help='Output of model prediction')
required.add_argument('-m', '--model', dest='model', type=str, required=True, choices=['HiCARN', 'DeepHiC', 'HiCNN', 'HiCSR', 'DFHiC', 'hicplus', 'SRHiC', 'iEnhance'])
required.add_argument('-o', '--output_dir', dest='output_dir', type=str, required=True, help='Directory path to save chromosome matrix')


args = parser.parse_args()
model = args.model.split()
input_data = args.input_data.split()
output_dir = args.output_dir.split()
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
        mats[str(chrom)] = np.triu(mat) + np.triu(mat, k=1).T # mats[str(chrom)] = mat
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
        mats[str(chrom)] = np.triu(mat) + np.triu(mat, k=1).T # mats[str(chrom)] = mat
    np.savez_compressed(save_filename, **mats)

        
if np.squeeze(np.load(input_data, allow_pickle=True)['data']).shape[-1] == 40:
    make_whole_40(input_data, output_dir+'/'+os.listdir(input_data).split('/')[-1]+'_wholeMats')
    print(f'{output_dir+os.listdir(input_data).split('/')[-1]+'_wholeMats'} was done.')
elif np.squeeze(np.load(input_data, allow_pickle=True)['data']).shape[-1] == 28:
    make_whole_28(input_data, output_dir+'/'+os.listdir(input_data).split('/')[-1]+'_wholeMats')
    print(f'{output_dir+os.listdir(input_data).split('/')[-1]+'_wholeMats'} was done.')
else:
    print("The output of iEnhance doesn't need to create a chromosome matrix; it's already done within the output of the model.")