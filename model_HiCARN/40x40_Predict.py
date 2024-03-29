import sys
import time
import multiprocessing
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch
import Models.DeepHiC as DeepHiC
import Models.HiCARN_1 as HiCARN_1
import Models.HiCARN_2 as HiCARN_2
from Utils.io import spreadM, together
from Arg_Parser import *

###########################################################
root_dir = "/data/HiHiC-main"
cell_line = "GM12878"
low_res = "16"
ckpt_file = "/data/HiHiC-main/checkpoints_HiCARN2/00080_2.25.50_15.711"
# cuda = "0"
device = "cpu"
model = "HiCARN_1"
HiCARN_file = "/data/HiHiC-main/data_HiCARN/test/test_ratio16.npz"
hicarn_data = np.load(HiCARN_file, allow_pickle=True)
out_dir = "/data/HiHiC-main/output"
os.makedirs(out_dir, exist_ok=True)
###########################################################

def dataloader(data, batch_size=64):
    inputs = torch.tensor(data['data'], dtype=torch.float)
    target = torch.tensor(data['target'], dtype=torch.float)
    inds = torch.tensor(data['inds'], dtype=torch.long)
    dataset = TensorDataset(inputs, target, inds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def data_info(data):
    indices = data['inds']
    compacts = data['compacts'][()]
    sizes = data['sizes'][()]
    return indices, compacts, sizes


get_digit = lambda x: int(''.join(list(filter(str.isdigit, x))))


def filename_parser(filename):
    info_str = filename.split('.')[0].split('_')[2:-1]
    chunk = get_digit(info_str[0])
    stride = get_digit(info_str[1])
    bound = get_digit(info_str[2])
    scale = 1 if info_str[3] == 'nonpool' else get_digit(info_str[3])
    return chunk, stride, bound, scale


def hicarn_predictor(model, hicarn_loader, ckpt_file, device):
    print(model)
    deepmodel = model.Generator(num_channels=64).to(device)
    if not os.path.isfile(ckpt_file):
        ckpt_file = f'save/{ckpt_file}'
    deepmodel.load_state_dict(torch.load(ckpt_file, map_location=torch.device('cpu')))
    print(f'Loading HiCARN checkpoint file from "{ckpt_file}"')

    result_data = []
    result_inds = []

    deepmodel.eval()
    with torch.no_grad():
        for batch in tqdm(hicarn_loader, desc='HiCARN Predicting: '):
            lr, hr, inds = batch
            lr = lr.to(device)
            out = deepmodel(lr)

            result_data.append(out.to('cpu').numpy())
            result_inds.append(inds.numpy())
    result_data = np.concatenate(result_data, axis=0)
    result_inds = np.concatenate(result_inds, axis=0)

    # hicarn_hics = together(result_data, result_inds, tag='Reconstructing: ')
    # return hicarn_hics
    return result_data, result_inds


# def save_data(carn, compact, size, file):
#     carn = spreadM(carn, compact, size, convert_int=False, verbose=True)
#     np.savez_compressed(file, hicarn=carn, compact=compact)
#     print('Saving file:', file)


if __name__ == '__main__':
    # args = data_predict_parser().parse_args(sys.argv[1:])
    # cell_line = args.cell_line
    # low_res = args.low_res
    # ckpt_file = args.checkpoint
    # cuda = args.cuda
    # model = args.model
    # HiCARN_file = args.file_name
    print('WARNING: Predict process requires large memory, thus ensure that your machine has ~150G memory.')
    # if multiprocessing.cpu_count() > 23:
    #     pool_num = 23
    # else:
    #     exit()

    # in_dir = os.path.join(root_dir, 'data')
    # out_dir = os.path.join(root_dir, 'predict', cell_line)
    # mkdir(out_dir)
    
    # files = [f for f in os.listdir(in_dir) if f.find(low_res) >= 0]

    # chunk, stride, bound, scale = filename_parser(HiCARN_file)

    # device = torch.device(
    #     f'cuda:{cuda}' if (torch.cuda.is_available() and cuda > -1 and cuda < torch.cuda.device_count()) else 'cpu')
    # print(f'Using device: {device}')

    start = time.time()
    print(f'Loading data[HiCARN]: {HiCARN_file}')
    # hicarn_data = np.load(os.path.join(in_dir, HiCARN_file), allow_pickle=True)
    hicarn_loader = dataloader(hicarn_data)

    indices, compacts, sizes = data_info(hicarn_data)

    if model == "HiCARN_1":
        model = HiCARN_1

    if model == "HiCARN_2":
        model = HiCARN_2

    if model == "DeepHiC":
        model = DeepHiC

    # hicarn_hics = hicarn_predictor(model, hicarn_loader, ckpt_file, device)
    result_data, result_inds = hicarn_predictor(model, hicarn_loader, ckpt_file, device)


    # def save_data_n(key):
    #     file = os.path.join(out_dir, f'predict_chr{key}_{low_res}.npz')
    #     save_data(hicarn_hics[key], compacts[key], sizes[key], file)
    # for key in sorted(list(np.unique(indices[:, 0]))):
    file = os.path.join(out_dir, f'HiCARN_predict_{low_res}.npz')
    np.savez_compressed(file, data=result_data, inds=result_inds)
    print('Saving file:', file)

    # pool = multiprocessing.Pool(processes=pool_num)
    # print(f'Start a multiprocess pool with process_num = {pool_num} for saving predicted data')
    # for key in compacts.keys():
    #     pool.apply_async(save_data_n, (key,))
    # pool.close()
    # pool.join()
    print(f'All data saved. Running cost is {(time.time() - start) / 60:.1f} min.')
