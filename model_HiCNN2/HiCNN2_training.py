import sys
import numpy as np
import pickle
import os
import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
#import model
from model import model1
from model import model2
from model import model3

##################################################################

import time, datetime

ROOT_DIR = './'
file_low_train = '/data/HiHiC-main/data_HiCNN/subMats_train_ratio16.npy'
file_high_train = '/data/HiHiC-main/data_HiCNN/subMats_train_target_ratio16.npy'
file_low_validate = '/data/HiHiC-main/data_HiCNN/subMats_valid_ratio16.npy'
file_high_validate = '/data/HiHiC-main/data_HiCNN/subMats_valid_target_ratio16.npy'
dir_models = os.path.join(ROOT_DIR, 'checkpoints_HiCNN2')
LOSS_LOG = 'train_loss_HiCNN2.npy'
batch_size = 256 # default
epochs = 500 # default

model = 3 # default
down_ratio = 16 # default
HiC_max = 100 # default
lr = 0.1 # default
momentum = 0.5 # default
weight_decay = 1e-4 # default
clip = 0.01 # default
no_cuda = False # default
seed = 1 # default

start = time.time()

train_epoch = [] 
train_loss = []
train_time = []

os.makedirs(dir_models, exist_ok=True)

##################################################################


# get current learning rate
def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']
# training 
def train(model, device, train_loader, optimizer, clip):
	model.train()
	loss_sum = 0.0
	for i, (data, target) in enumerate(train_loader):
		if i == (len(train_loader) - 1):
			continue
		data, target = Variable(data).to(device), Variable(target, requires_grad=False).to(device)
		optimizer.zero_grad()
		output = model(data)
		# loss function
		criterion = nn.MSELoss()
		loss = criterion(output, target)
		loss.backward()
		lr_current = get_lr(optimizer)
		clip2 = clip/lr_current
		nn.utils.clip_grad_norm_(model.parameters(),clip2)
		optimizer.step()
		loss_sum = loss_sum + loss.item()
	return loss_sum/i
# validation
def validate(model, device, validation_loader):
	model.eval()
	loss_sum = 0.0
	with torch.no_grad():
		for i, (data, target) in enumerate(validation_loader):
			data, target = Variable(data).to(device), Variable(target, requires_grad=False).to(device)
			output = model(data)
			# loss function
			criterion = nn.MSELoss()
			loss = criterion(output, target)
			loss_sum = loss_sum + loss.item()
	return loss_sum/i

def main():
	# parser = argparse.ArgumentParser(description='HiCNN2 training process')
	# parser._action_groups.pop()
	# required = parser.add_argument_group('required arguments')
	# optional = parser.add_argument_group('optional arguments')
	# required.add_argument('-f1', '--file-training-data', type=str, metavar='FILE', required=True,
    #                     help='file name of the training data, npy format and shape=n1*1*40*40')
	# required.add_argument('-f2', '--file-training-target', type=str, metavar='FILE', required=True,
    #                     help='file name of the training target, npy format and shape=n1*1*40*40')
	# required.add_argument('-f3', '--file-validate-data', type=str, metavar='FILE', required=True,
    #                     help='file name of the validation data, npy format and shape=n2*1*40*40')
	# required.add_argument('-f4', '--file-validate-target', type=str, metavar='FILE', required=True,
    #                     help='file name of the validation target, npy format and shape=n2*1*40*40')
	# required.add_argument('-m', '--model', type=int, default=3, metavar='N', required=True,
	#                       help='1:HiCNN2-1, 2:HiCNN2-2, and 3:HiCNN2-3 (default: 3)')
	# required.add_argument('-d', '--dir-models', type=str, metavar='DIR', required=True,
    #                     help='directory for saving models')
	# required.add_argument('-r', '--down-ratio', type=int, default=16, metavar='N', required=True,
    #                     help='down sampling ratio, 16 means 1/16 (default: 16)')
	# optional.add_argument('--HiC-max', type=int, default=100, metavar='N',
    #                     help='the maximum value of Hi-C contacts (default: 100)')
	# optional.add_argument('--batch-size', type=int, default=256, metavar='N',
    #                     help='input batch size for training (default: 256)')
	# optional.add_argument('--epochs', type=int, default=500, metavar='N',
    #                     help='number of epochs to train (default: 500)')
	# optional.add_argument('--lr', type=float, default=0.1, metavar='LR',
    #                     help='initial learning rate (default: 0.1)')
	# optional.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                     help='SGD momentum (default: 0.5)')
	# optional.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
    #                     help='weight-decay (default: 1e-4)')
	# optional.add_argument('--clip', type=float, default=0.01, metavar='M',
    #                     help='clip (default: 0.01)')
	# optional.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')	
	# optional.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')	
	# args = parser.parse_args()
 

	torch.manual_seed(seed)

	low_res_train = np.minimum(HiC_max, np.load(file_low_train).astype(np.float32) * down_ratio)
	high_res_train = np.minimum(HiC_max, np.load(file_high_train).astype(np.float32))
	low_res_validate = np.minimum(HiC_max, np.load(file_low_validate).astype(np.float32) * down_ratio)
	high_res_validate = np.minimum(HiC_max, np.load(file_high_validate).astype(np.float32))

	target_train = []
	for i in range(high_res_train.shape[0]):
		target_train.append([high_res_train[i][0][6:34, 6:34],])
	target_train = np.array(target_train).astype(np.float32)
	target_validate = []
	for i in range(high_res_validate.shape[0]):
		target_validate.append([high_res_validate[i][0][6:34, 6:34],])
	target_validate = np.array(target_validate).astype(np.float32)

	train_loader = torch.utils.data.DataLoader(data.TensorDataset(torch.from_numpy(low_res_train), torch.from_numpy(target_train)), batch_size=batch_size, shuffle=True)
	validation_loader = torch.utils.data.DataLoader(data.TensorDataset(torch.from_numpy(low_res_validate), torch.from_numpy(target_validate)), batch_size=batch_size, shuffle=True)

	# check if CUDA is available
	use_cuda = not no_cuda and torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
	global model ######################
	if model == 1:
		print("Using HiCNN2-1...")
		model = model1.Net().to(device)
	elif model == 2:
		print("Using HiCNN2-2...")
		model = model2.Net().to(device)
	else:
		print("Using HiCNN2-3...")
		model = model3.Net().to(device)
	
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
	# reducing learning rate when loss from validation has stopped improving
	scheduler = ReduceLROnPlateau(optimizer, 'min')
	for epoch in range(1, epochs+1):
		loss_train = train(model, device, train_loader, optimizer, clip) 
		loss_validate = validate(model, device, validation_loader)
		scheduler.step(loss_validate)	
		lr_current = get_lr(optimizer)
		print(epoch, lr_current, loss_train, loss_validate, datetime.datetime.now())  
		# save the model
		# torch.save(model.state_dict(), dir_models + '/'  + str(epoch))

		##################################################################
        
		if epoch%10 == 0:
			sec = time.time()-start
			times = str(datetime.timedelta(seconds=sec))
			short = times.split(".")[0].replace(':','.')
				
			train_epoch.append(epoch)
			train_time.append(short)        
			train_loss.append(f"{loss_validate:.2f}")
			
			ckpt_file = f"{str(epoch).zfill(5)}_{short}"
			torch.save(model.state_dict(), os.path.join(dir_models, ckpt_file))
		
	np.save(LOSS_LOG, [train_epoch, train_time, train_loss])
 
		##################################################################


if __name__ == '__main__':
	main()