import os,sys
from torch.utils import data
# from HiCPlus import model
import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# import straw
from scipy.sparse import csr_matrix, coo_matrix, vstack, hstack
from scipy import sparse
import numpy as np
# from HiCPlus import utils
from time import gmtime, strftime
from datetime import datetime
import argparse


################################################## Added by HiHiC ######
########################################################################

parser = argparse.ArgumentParser(description='HiCPlus prediction process')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')

required.add_argument('--root_dir', type=str, metavar='/HiHiC', required=True,
                      help='HiHiC directory')
required.add_argument('--model', type=str, default='HiCPlus', metavar='HiCPlus', required=True,
                      help='model name')
required.add_argument('--ckpt_file', type=str, metavar='[2]', required=True,
                      help='pretrained model')
required.add_argument('--batch_size', type=int, default=64, metavar='[3]', required=True,
                      help='input batch size for training (default: 64)')
required.add_argument('--gpu_id', type=int, default=0, metavar='[4]', required=True, 
                      help='GPU ID for training (defalut: 0)')
required.add_argument('--down_ratio', type=int, metavar='[5]', required=True, 
                      help='down sampling ratio')
required.add_argument('--input_data', type=str, metavar='[6]', required=True,
                      help='directory path of training model')
required.add_argument('--output_data_dir', type=str, default='./output_enhanced', metavar='[7]', required=True,
                      help='directory path for saving enhanced output (default: HiHiC/output_enhanced/)')
args = parser.parse_args()

os.makedirs(args.output_data_dir, exist_ok=True) #######################
########################################################################


startTime = datetime.now()

# use_gpu = True #opt.cuda
#if use_gpu and not torch.cuda.is_available():
#    raise Exception("No GPU found, please run without --cuda")
use_gpu = torch.cuda.is_available() ##############################################################
if args.gpu_id == -1:
    device  = torch.device('cpu')  # CPU 사용
else:
    device  = torch.device(f'cuda:{args.gpu_id}')  # GPU 사용 ###################### Added by HiHiC

def predict(M,inmodel):
    # N = M['data'].shape[0]

    # prediction_1 = np.zeros((N, N))
    y_predict = []

    # for low_resolution_samples, index in utils.divide(M):
    for low_resolution_samples in M['data']:

        #print(index.shape)

        # batch_size = low_resolution_samples.shape[0] #256

        lowres_set = data.TensorDataset(torch.from_numpy(low_resolution_samples), torch.from_numpy(np.zeros(low_resolution_samples.shape[0])))
        try:
            lowres_loader = torch.utils.data.DataLoader(lowres_set, batch_size=args.batch_size, shuffle=False)
        except:
            continue

        # hires_loader = lowres_loader

        m = model.Net(40, 28)
        # m.load_state_dict(torch.load(inmodel, map_location=torch.device('cpu')))
        # m.load_state_dict(torch.load(inmodel, map_location=torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')))
        m.load_state_dict(torch.load(inmodel, map_location=device))


        # if torch.cuda.is_available():
        #     m = m.cuda()
        if use_gpu:
            m =m.to(device)

        for i, v1 in enumerate(lowres_loader):
            _lowRes, _ = v1
            _lowRes = Variable(_lowRes).float()
            if use_gpu:
                # _lowRes = _lowRes.cuda()
                _lowRes = _lowRes.to(device)
            _lowRes = _lowRes.unsqueeze(0) ###################### Added by HiHiC
            y_prediction = m(_lowRes)

        # y_predict = y_prediction.data.cpu().numpy()
            y_predict.append(y_prediction.data.cpu().numpy())
    #     # recombine samples
    #     length = int(y_predict.shape[2])
    #     y_predict = np.reshape(y_predict, (y_predict.shape[0], length, length))


        # for i in range(0, y_predict.shape[0]):

        #     x = int(index[i][1])
        #     y = int(index[i][2])
        #     #print np.count_nonzero(y_predict[i])
        #     prediction_1[x+6:x+34, y+6:y+34] = y_predict[i]
    prediction_1 = np.concatenate(y_predict, axis=0)

    return(prediction_1)

# def chr_pred(hicfile, chrN1, chrN2, binsize, inmodel):
#     M = utils.matrix_extract(chrN1, chrN2, binsize, hicfile)
#     #print(M.shape)
#     N = M.shape[0]

#     chr_Mat = predict(M, N, inmodel)


# #     if Ncol > Nrow:
# #         chr_Mat = chr_Mat[:Ncol, :Nrow]
# #         chr_Mat = chr_Mat.T
# #     if Nrow > Ncol: 
# #         chr_Mat = chr_Mat[:Nrow, :Ncol]
# #     print(dat.head())       
#     return(chr_Mat)
HiCPlus_data = np.load(args.input_data, allow_pickle=True)


# def writeBed(Mat, outname,binsize, chrN1,chrN2):
#     with open(outname,'w') as chrom:
#         r, c = Mat.nonzero()
#         for i in range(r.size):
#             contact = int(round(Mat[r[i],c[i]]))
#             if contact == 0:
#                 continue
#             #if r[i]*binsize > Len1 or (r[i]+1)*binsize > Len1:
#             #    continue
#             #if c[i]*binsize > Len2 or (c[i]+1)*binsize > Len2:
#             #    continue
#             line = [chrN1, r[i]*binsize, (r[i]+1)*binsize,
#                chrN2, c[i]*binsize, (c[i]+1)*binsize, contact]
#             chrom.write('chr'+str(line[0])+':'+str(line[1])+'-'+str(line[2])+
#                      '\t'+'chr'+str(line[3])+':'+str(line[4])+'-'+str(line[5])+'\t'+str(line[6])+'\n')

# def main(args):
def main():
    # chrN1, chrN2 = args.chrN
    # binsize = args.binsize
    # inmodel = args.model
    # hicfile = args.inputfile
    # #name = os.path.basename(inmodel).split('.')[0]
    # #outname = 'chr'+str(chrN1)+'_'+name+'_'+str(binsize//1000)+'pred.txt'
    # outname = args.outputfile
    # Mat = chr_pred(hicfile,chrN1,chrN2,binsize,inmodel)
    # ckpt_file=args.ckpt_file
    Mat = predict(HiCPlus_data,args.ckpt_file)
    print(Mat.shape)
    # writeBed(Mat, outname, binsize,chrN1, chrN2)
    # for key in sorted(list(np.unique(hicarn_data['indz'][:, 0]))):
    th_model = args.ckpt_file.split('/')[-1].split('_')[0]
    file = os.path.join(args.output_data_dir, f'HiCPlus_predict_{args.down_ratio}_{th_model}.npz')
    np.savez_compressed(file, data=Mat, inds=HiCPlus_data['inds_target'])
    print('Saving file:', file)
if __name__ == '__main__':
    main()

print(datetime.now() - startTime)
