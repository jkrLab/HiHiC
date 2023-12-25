from __future__ import print_function
import argparse as ap
from math import log10

#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.autograd import Variable
#from torch.utils.data import DataLoader
# from hicplus import utils
#import model
import argparse
# from hicplus import trainConvNet
import trainConvNet
import numpy as np


##################################################################### Added by HiHiC ##
import os #############################################################################

parser = argparse.ArgumentParser(description='hicplus training process')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('--root_dir', type=str, metavar='/HiHiC', required=True,
                      help='HiHiC directory')
required.add_argument('--model', type=str, metavar='hicplus', required=True,
                      help='model name')
required.add_argument('--epoch', type=int, default=128, metavar='[2]', required=True,
                      help='training epoch (default: 128)')
required.add_argument('--batch_size', type=int, default=64, metavar='[3]', required=True,
                      help='input batch size for training (default: 64)')
required.add_argument('--gpu_id', type=int, default=0, metavar='[4]', required=True, 
                      help='GPU ID for training (defalut: 0)')
required.add_argument('--output_model_dir', type=str, default='./checkpoints_hicplus', metavar='[5]', required=True,
                      help='directory path of training model (default: HiHiC/checkpoints_hicplus/)')
required.add_argument('--loss_log_dir', type=str, default='./log', metavar='[6]', required=True,
                      help='directory path of training log (default: HiHiC/log/)')
required.add_argument('--train_data_dir', type=str, metavar='[7]', required=True,
                      help='directory path of training data')
optional.add_argument('--valid_data_dir', type=str, metavar='[8]',
                      help="directory path of validation data, but hicplus doesn't need")
args = parser.parse_args() ############################################################
#######################################################################################


# chrs_length = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]

#chrN = 21
#scale = 16

# def main(args):
def main():

    # highres = utils.train_matrix_extract(args.chromosome, 10000, args.inputfile) # each chromosome matrix by juicer
     
    # print('dividing, filtering and downsampling files...')

    # highres_sub, index = utils.train_divide(highres) # 40X40 submatrix and index (np.array)
    
    # print(highres_sub.shape)
    # np.save(infile+"highres",highres_sub)

    # lowres = utils.genDownsample(highres,1/float(args.scalerate))
    # lowres_sub,index = utils.train_divide(lowres) # 40X40 submatrix and index (np.array)
    # print(lowres_sub.shape)
    # np.save(infile+"lowres",lowres_sub)
    
    data_all = [np.load(os.path.join(args.train_data_dir, fname), allow_pickle=True) for fname in os.listdir(args.train_data_dir)] ### Added by HiHiC ##
    # print(data_all)
    # print(data_all[0].files, "#####################")
    train = {'data': [], 'target': [], 'inds_data': [], 'inds_target': []} #####################################################################################################
    for data in data_all:
        for k, v in data.items():
            if k in train: 
                train[k].append(v) #####################################################################################################################
    train = {k: np.concatenate(v, axis=0) for k, v in train.items()} ###################################################################################
 

    print('start training...')
    # trainConvNet.train(lowres_sub,highres_sub,args.outmodel)
    trainConvNet.train(train['data'],train['target'],args.output_model_dir,args.epoch,args.batch_size,args.gpu_id, args.loss_log_dir)

    print('finished...')

main()