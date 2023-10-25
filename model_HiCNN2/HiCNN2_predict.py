import sys
import numpy as np
import pickle
#import model
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from model import model1
from model import model2
from model import model3

parser = argparse.ArgumentParser(description='HiCNN2 predicting process')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('-f1', '--file-test-data', type=str, metavar='FILE', required=True,
                        help='file name of the test data, npy format and shape=n1*1*40*40')
required.add_argument('-f2', '--file-test-predicted', type=str, metavar='FILE', required=True,
                        help='file name to save the predicted target, npy format and shape=n1*1*28*28')
required.add_argument('-mid', '--model', type=int, default=3, metavar='N', required=True,
                        help='1:HiCNN2-1, 2:HiCNN2-2, and 3:HiCNN2-3 (default: 3)')
required.add_argument('-m', '--file-best-model', type=str, metavar='FILE', required=True,
                        help='file name of the best model')
required.add_argument('-r', '--down-ratio', type=int, default=16, metavar='N', required=True,
                        help='down sampling ratio, 16 means 1/16 (default: 16)')
optional.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA predicting')
optional.add_argument('--HiC-max', type=int, default=100, metavar='N',
                        help='the maximum value of Hi-C contacts (default: 100)')
optional.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for test (default: 128)')	
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if args.model == 1:
	print("Using HiCNN2-1...")
	Net = model1.Net().to(device).eval()
elif args.model == 2:
	print("Using HiCNN2-2...")
	Net = model2.Net().to(device).eval()
else:
	print("Using HiCNN2-3...")
	Net = model3.Net().to(device).eval()

Net.load_state_dict(torch.load(args.file_best_model))

low_res_test = np.minimum(args.HiC_max, np.load(args.file_test_data).astype(np.float32) * args.down_ratio)
test_loader = torch.utils.data.DataLoader(data.TensorDataset(torch.from_numpy(low_res_test), torch.from_numpy(np.zeros(low_res_test.shape[0]))), batch_size=args.batch_size, shuffle=False)

result = np.zeros((low_res_test.shape[0],1,28,28))
for i, (data, _) in enumerate(test_loader):
	data2 = Variable(data).to(device)
	output = Net(data2)
	resulti = output.cpu().data.numpy()
	resulti = np.squeeze(resulti)
	i1 = i * args.batch_size
	i2 = i1 + args.batch_size
	if i == int(low_res_test.shape[0]/args.batch_size):
		i2 = low_res_test.shape[0]
	result[i1:i2,0,:,:] = resulti

np.save(args.file_test_predicted, result)


