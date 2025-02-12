# Author: Yan Zhang
# Email: zhangyan.cse (@) gmail.com

import sys
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import os
import gzip
# from HiCPlus 
import model
from torch.utils import data
import torch
import torch.optim as optim
from torch.autograd import Variable
from time import gmtime, strftime
import sys
import torch.nn as nn
import argparse
import os, time, datetime


# use_gpu = 1

conv2d1_filters_numbers = 8
conv2d1_filters_size = 9
conv2d2_filters_numbers = 8
conv2d2_filters_size = 1
conv2d3_filters_numbers = 1
conv2d3_filters_size = 5


# down_sample_ratio = 16
# epochs = 10
# HiC_max_value = 100
# batch_size = 512


# This block is the actual training data used in the training. The training data is too large to put on Github, so only toy data is used.
# cell = "GM12878_replicate"
# chrN_range1 = '1_8'
# chrN_range = '1_8'

# low_resolution_samples = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/'+cell+'down16_chr'+chrN_range+'.npy.gz', "r")).astype(np.float32) * down_sample_ratio
# high_resolution_samples = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/original10k/'+cell+'_original_chr'+chrN_range+'.npy.gz', "r")).astype(np.float32)

# low_resolution_samples = np.minimum(HiC_max_value, low_resolution_samples)
# high_resolution_samples = np.minimum(HiC_max_value, high_resolution_samples)


#low_resolution_samples = np.load(gzip.GzipFile('../../data/GM12878_replicate_down16_chr19_22.npy.gz', "r")).astype(np.float32) * down_sample_ratio
#high_resolution_samples = np.load(gzip.GzipFile('../../data/GM12878_replicate_original_chr19_22.npy.gz', "r")).astype(np.float32)

#low_resolution_samples = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/IMR90_down_HINDIII16_chr1_8.npy.gz', "r")).astype(np.float32) * down_sample_ratio
#high_resolution_samples = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/original10k/_IMR90_HindIII_original_chr1_8.npy.gz', "r")).astype(np.float32)

# def train(lowres,highres, outModel):
def train(lowres, highres, outModel, EPOCH, BATCH_SIZE, GPU_ID, LOSS_LOG_DIR):
    
    ########################################################## Added by HiHiC ##
    ############################################################################
    start = time.time()

    train_epoch = [] 
    train_loss = []
    train_time = []

    os.makedirs(outModel, exist_ok=True)
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    ############################################################################
    ############################################################################
    
    # low_resolution_samples = lowres.astype(np.float32) * down_sample_ratio
    # high_resolution_samples = highres.astype(np.float32)
    low_resolution_samples = lowres.astype(np.float32)
    high_resolution_samples = highres.astype(np.float32)

    # low_resolution_samples = np.minimum(HiC_max_value, low_resolution_samples)
    # high_resolution_samples = np.minimum(HiC_max_value, high_resolution_samples)

    # Reshape the high-quality Hi-C sample as the target value of the training.
    sample_size = low_resolution_samples.shape[-1]
    padding = conv2d1_filters_size + conv2d2_filters_size + conv2d3_filters_size - 3
    half_padding = padding // 2
    output_length = sample_size - padding
    Y = []
    for i in range(high_resolution_samples.shape[0]):
        no_padding_sample = high_resolution_samples[i][0][half_padding:(sample_size-half_padding) , half_padding:(sample_size - half_padding)]
        Y.append(no_padding_sample)
    Y = np.array(Y).astype(np.float32)

    print(low_resolution_samples.shape, Y.shape)

    lowres_set = data.TensorDataset(torch.from_numpy(low_resolution_samples), torch.from_numpy(np.zeros(low_resolution_samples.shape[0])))
    lowres_loader = torch.utils.data.DataLoader(lowres_set, batch_size=BATCH_SIZE, shuffle=False)

    hires_set = data.TensorDataset(torch.from_numpy(Y), torch.from_numpy(np.zeros(Y.shape[0])))
    hires_loader = torch.utils.data.DataLoader(hires_set, batch_size=BATCH_SIZE, shuffle=False)


    Net = model.Net(40, 28)

    # if use_gpu:
        # Net = Net.cuda()    
    Net = Net.to(device)

    optimizer = optim.SGD(Net.parameters(), lr = 0.00001)
    _loss = nn.MSELoss()
    ########################################################## Added by HiHiC #####
    Net.eval() ####################################################################
    loss_sum = 0.0

    with torch.no_grad():
        # 전체 데이터를 미니배치 단위로 처리
        for (data_batch, _), (target_batch, _) in zip(lowres_loader, hires_loader):  # lowres_loader와 hires_loader의 데이터를 동시에 가져옴
            data_batch = data_batch.to(device)  # Move lowres data to device
            target_batch = target_batch.unsqueeze(1).to(device)  # Move target data to device
            output = Net(data_batch)
            loss = _loss(output, target_batch)
            loss_sum += loss.item()

    initial_train_loss = loss_sum / len(lowres_loader)

    train_epoch.append(int(0))
    train_time.append("0.00.00")
    train_loss.append(f"{initial_train_loss:.10f}")
    np.save(os.path.join(LOSS_LOG_DIR, f'train_loss_HiCPlus'), [train_epoch, train_time, train_loss])
    ###############################################################################
    ###############################################################################
    Net.train()

    running_loss = 0.0
    running_loss_validate = 0.0
    reg_loss = 0.0

    # write the log file to record the training process
    # with open('HindIII_train.txt', 'w') as log:
    # for epoch in range(0, 3500):
    for epoch in range(1, int(EPOCH)+1):
        for i, (v1, v2) in enumerate(zip(lowres_loader, hires_loader)):
            if (i == len(lowres_loader) - 1):
                continue
            _lowRes, _ = v1
            _highRes, _ = v2

            _lowRes = Variable(_lowRes)
            _highRes = Variable(_highRes).unsqueeze(1)

            # if use_gpu:
            #     _lowRes = _lowRes.cuda()
            #     _highRes = _highRes.cuda()
            _lowRes = _lowRes.to(device)
            _highRes = _highRes.to(device)
            optimizer.zero_grad()
            y_prediction = Net(_lowRes)

            loss = _loss(y_prediction, _highRes)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()

        if epoch: #################################################### Added by HiHiC ##  
            sec = time.time()-start
            times = str(datetime.timedelta(seconds=sec))
            short = times.split(".")[0].replace(':','.')
                
            train_epoch.append(epoch)
            train_time.append(short)        
            train_loss.append(f"{loss:.10f}")
            
            ckpt_file = f"{str(epoch).zfill(5)}_{short}_{loss:.10f}"
            torch.save(Net.state_dict(), os.path.join(outModel, ckpt_file))
            np.save(os.path.join(LOSS_LOG_DIR, f'train_loss_HiCPlus'), [train_epoch, train_time, train_loss])            
        ##############################################################################
        ##############################################################################

    print('-------', i, epoch, running_loss/i, strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    # log.write(str(epoch) + ', ' + str(running_loss/i,) +', '+ strftime("%Y-%m-%d %H:%M:%S", gmtime())+ '\n')
    running_loss = 0.0
    running_loss_validate = 0.0
	# save the model every 100 epoches
        # if (epoch % 100 == 0):
        #     torch.save(Net.state_dict(), outModel + str(epoch) + str('.model'))
        # pass
    
     ### Added by HiHiC ##
    