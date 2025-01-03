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


################################################## Added by HiHiC ######
########################################################################

import argparse

parser = argparse.ArgumentParser(description='HiCARN prediction process')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')

required.add_argument('--root_dir', type=str, metavar='/HiHiC', required=True,
                      help='HiHiC directory')
required.add_argument('--model', type=str, default='HiCARN2', metavar='HiCARN', required=True,
                      help='model name')
required.add_argument('--ckpt_file', type=str, metavar='[2]', required=True,
                      help='pretrained model')
required.add_argument('--batch_size', type=int, default=64, metavar='[3]', required=True,
                      help='input batch size for training (default: 64)')
required.add_argument('--gpu_id', type=int, default=0, metavar='[4]', required=True, 
                      help='GPU ID for training (defalut: 0)')
required.add_argument('--input_data', type=str, metavar='[5]', required=True,
                      help='directory path of training model')
required.add_argument('--output_data_dir', type=str, default='./output_enhanced', metavar='[6]', required=True,
                      help='directory path for saving enhanced output (default: HiHiC/output_enhanced/)')
args = parser.parse_args()

if args.model == "HiCANR1":
    model = "HiCARN_1"
else:
    model = "HiCARN_2"

prefix = os.path.splitext(os.path.basename(args.input_data))[0]
os.makedirs(args.output_data_dir, exist_ok=True) #######################
########################################################################


def dataloader(data, batch_size=64):
    inputs = torch.tensor(data['data'], dtype=torch.float)
    # target = torch.tensor(data['target'], dtype=torch.float)
    inds = torch.tensor(data['inds'], dtype=torch.long)
    # dataset = TensorDataset(inputs, target, inds)
    dataset = TensorDataset(inputs, inds)
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
            # lr, hr, inds = batch
            lr, inds = batch
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

    device = torch.device(
        f'cuda:{args.gpu_id}' if (torch.cuda.is_available() and int(args.gpu_id) > -1 and int(args.gpu_id) < torch.cuda.device_count()) else 'cpu')
    print(f'Using device: {device}')

    start = time.time()
    print(f'Loading data[HiCARN]: {args.input_data}')
    # hicarn_data = np.load(os.path.join(in_dir, HiCARN_file), allow_pickle=True)
    hicarn_data = np.load(args.input_data, allow_pickle=True)
    hicarn_loader = dataloader(hicarn_data)

    indices, compacts, sizes = data_info(hicarn_data)

    if model == "HiCARN_1":
        model = HiCARN_1

    if model == "HiCARN_2":
        model = HiCARN_2

    if model == "DeepHiC":
        model = DeepHiC

    # hicarn_hics = hicarn_predictor(model, hicarn_loader, ckpt_file, device)
    result_data, result_inds = hicarn_predictor(model, hicarn_loader, args.ckpt_file, device)


    # def save_data_n(key):
    #     file = os.path.join(out_dir, f'predict_chr{key}_{low_res}.npz')
    #     save_data(hicarn_hics[key], compacts[key], sizes[key], file)
    # for key in sorted(list(np.unique(indices[:, 0]))):
    th_model = args.ckpt_file.split('/')[-1].split('_')[0]
    file = os.path.join(args.output_data_dir, f'{prefix}_{args.model}_{th_model}ep.npz')
    np.savez_compressed(file, data=result_data, inds=result_inds)
    print('Saving file:', file)

    # pool = multiprocessing.Pool(processes=pool_num)
    # print(f'Start a multiprocess pool with process_num = {pool_num} for saving predicted data')
    # for key in compacts.keys():
    #     pool.apply_async(save_data_n, (key,))
    # pool.close()
    # pool.join()
    print(f'All data saved. Running cost is {(time.time() - start) / 60:.1f} min.')
