import numpy as np


#########################################################
import os, random, math, argparse
from sklearn.preprocessing import MinMaxScaler
random.seed(100)
scaler = MinMaxScaler(feature_range=(0,1))

# 인자 받기
parser = argparse.ArgumentParser(description='Read Hi-C contact map and Divide submatrix for train and predict', add_help=True)
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('-i', '--input_data_dir', dest='input_data_dir', type=str, required=True, help='Input data directory: /HiHiC/data')
required.add_argument('-m', '--model', dest='model', type=str, required=True, choices=['HiCARN', 'DeepHiC', 'HiCNN', 'HiCSR', 'DFHiC', 'HiCPlus', 'SRHiC', 'iEnhance'])
required.add_argument('-r', '--data_ratio', dest='data_ratio', type=str, required=True, help='Downsampling ratio: 16')
required.add_argument('-o', '--output_dir', dest='output_dir', type=str, required=True, help='Parent directory of output: /data/HiHiC/')
required.add_argument('-s', '--max_value', dest='max_value', type=str, required=True, default='300', help='Maximum value of chromosome matrix')
required.add_argument('-n', '--normalization', dest='normalization', type=str, required=True, default='None', help='Normalization method')
optional.add_argument('-t', '--train_set', dest='train_set', type=str, required=False, help='Train set chromosome: "1 2 3 4 5 6 7 8 9 10 11 12 13 14"')
optional.add_argument('-v', '--valid_set', dest='valid_set', type=str, required=False, help='Validation set chromosome: "15 16 17"')
optional.add_argument('-p', '--test_set', dest='test_set', type=str, required=False, help='Prediction set chromosome: "18 19 20 21 22"')

args = parser.parse_args()

normalization = args.normalization
max_value = args.max_value # minmax scaling
save_dir = f'{args.output_dir}/data_{args.model}/'
os.makedirs(save_dir, exist_ok=True)

train_dir = save_dir + f"train_{normalization}_{max_value}/"
valid_dir = save_dir + f"valid_{normalization}_{max_value}/"
test_dir = save_dir + f"test_{normalization}_{max_value}/"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
#########################################################

# deal_fd = "data/"
# fn = 'gm64'
# trainlist = ['2']
# testlist = ['1']
deal_fd = args.input_data_dir
fn = args.output_dir
trainlist = args.train_set.split()
testlist = args.test_set.split()
validlist = args.valid_set.split()

lrtraind = []
# lrtestd = []
hrtraind = []
# hrtestd = []

for c in trainlist:
    datashr = np.load(deal_fd + 'chr' + c + '.npz')['target']
    dataslr = np.load(deal_fd + 'chr' + c + '.npz')['data']

    bnum = dataslr.shape[0]
    idx = np.random.choice(np.arange(bnum),int(bnum/2),replace=False)
    lrout = dataslr[idx,...]
    hrout = datashr[idx,...]

    lrtraind.append(lrout)
    hrtraind.append(hrout)

lrtraind = np.concatenate(lrtraind,axis=0)
hrtraind = np.concatenate(hrtraind,axis=0)
# np.savez(fn + "_train.npz",hr_sample = hrtraind,lr_sample = lrtraind)
np.savez(train_dir + f"train_ratio{args.data_ratio}.npz",target = hrtraind,data = lrtraind)

# for c in testlist:
#     datashr = np.load(deal_fd + 'chr-' + c + '.npz')['hr_sample']
#     dataslr = np.load(deal_fd + 'chr-' + c + '.npz')['lr_sample']

#     bnum = dataslr.shape[0]
#     idx = np.random.choice(np.arange(bnum),int(bnum/2),replace=False)
#     lrout = dataslr[idx,...]
#     hrout = datashr[idx,...]

#     lrtestd.append(lrout)
#     hrtestd.append(hrout)

# lrtestd = np.concatenate(lrtestd,axis=0)
# hrtestd = np.concatenate(hrtestd,axis=0)
# np.savez(fn + "_test.npz",hr_sample = hrtestd,lr_sample = lrtestd)

############################################################################ by HiHiC #########

lrvalid = []
hrvalid = []

for c in validlist:
    datashr = np.load(deal_fd + 'chr' + c + '.npz')['target']
    dataslr = np.load(deal_fd + 'chr' + c + '.npz')['data']

    bnum = dataslr.shape[0]
    idx = np.random.choice(np.arange(bnum),int(bnum/2),replace=False)
    lrout = dataslr[idx,...]
    hrout = datashr[idx,...]

    lrvalid.append(lrout)
    hrvalid.append(hrout)

lrvalid = np.concatenate(lrvalid,axis=0)
hrvalid = np.concatenate(hrvalid,axis=0)
# np.savez(fn + "_test.npz",hr_sample = hrtestd,lr_sample = lrtestd)
np.savez(valid_dir + f"valid_ratio{args.data_ratio}.npz",target = hrvalid,data = lrvalid)


whole_hrmat = {}
whole_lrmat = {}

for c in testlist:

    datashr = np.load(deal_fd + 'chr' + c + '_whole_mat.npz')['target']
    dataslr = np.load(deal_fd + 'chr' + c + '_whole_mat.npz')['data']
    
    whole_hrmat[c] = datashr
    whole_lrmat[c] = dataslr

np.savez(test_dir + f"test_ratio{args.data_ratio}_target.npz", **whole_hrmat)
np.savez(test_dir + f"test_ratio{args.data_ratio}.npz", **whole_lrmat)

print(f"train set: {trainlist} \nvalid set: {validlist} \ntest set: {testlist}" , flush=True)

###################################################################################################