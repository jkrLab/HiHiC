from __future__ import print_function
import argparse as ap
from math import log10

#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.autograd import Variable
#from torch.utils.data import DataLoader
# from hicplus import utils # 가져다 쓴 util 없어서 주석처리
#import model
# import argparse
# from hicplus 
import trainConvNet
import numpy as np


##################################################################

import os

ROOT_DIR = './'
OUT_DIR = os.path.join(ROOT_DIR, 'checkpoints_hicplus')
TRAIN_DATA = '/data/HiHiC-main/data_hicplus/subMats_train_ratio16.npy'
TRAIN_TARGET = '/data/HiHiC-main/data_hicplus/subMats_train_target_ratio16.npy'

##################################################################

# chrs_length = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]

#chrN = 21
#scale = 16

# def main(args):
def main():

    # highres = utils.train_matrix_extract(args.chromosome, 10000, args.inputfile) # each chromosome matrix by juicer
     
    # print('dividing, filtering and downsampling files...')

    # highres_sub, index = utils.train_divide(highres) # 40X40 submatrix and index (np.array)
    highres_sub = np.load(TRAIN_TARGET)

    print(highres_sub.shape)
    #np.save(infile+"highres",highres_sub)

    # lowres = utils.genDownsample(highres,1/float(args.scalerate))
    # lowres_sub,index = utils.train_divide(lowres) # 40X40 submatrix and index (np.array)
    lowres_sub = np.load(TRAIN_DATA)
    
    print(lowres_sub.shape)
    #np.save(infile+"lowres",lowres_sub)

    print('start training...')
    trainConvNet.train(lowres_sub,highres_sub,OUT_DIR)

    print('finished...')

main()