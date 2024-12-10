import numpy as np
import multiprocessing
import torch as t
import time
# import cooler

from normga4 import Construct
from model import iEnhance


################################################## Added by HiHiC ######
########################################################################
import os, argparse

parser = argparse.ArgumentParser(description='iEnhance prediction process')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')

required.add_argument('--root_dir', type=str, metavar='/HiHiC', required=True,
                      help='HiHiC directory')
required.add_argument('--model', type=str, default='iEnhance', metavar='iEnhance', required=True,
                      help='model name')
required.add_argument('--ckpt_file', type=str, metavar='[2]', required=True,
                      help='pretrained model')
required.add_argument('--batch_size', type=int, default=64, metavar='[3]', required=True,
                      help='input batch size for training (default: 64)')
required.add_argument('--gpu_id', type=int, default=0, metavar='[4]', required=True, 
                      help='GPU ID for training (defalut: 0)')
required.add_argument('--input_data', type=str, metavar='[5]', required=True,
                      help='directory path of training model')
required.add_argument('--output_data_dir', type=str, default='./output_iEnhance', metavar='[6]', required=True,
                      help='directory path for saving enhanced output (default: HiHiC/output_iEnhance/)')
args = parser.parse_args()

model = Construct() 
state_dict = t.load(args.ckpt_file, map_location = t.device('cpu'))
model.load_state_dict(state_dict)

device = t.device(f'cuda:{args.gpu_id}' if t.cuda.is_available() else 'cpu')
model.to(device)

prefix = os.path.splitext(os.path.basename(args.input_data))[0]
input_data = dict(np.load(args.input_data, allow_pickle=True))
chrs_list = input_data.keys()

########################################################################
########################################################################

# model = t.load("pretrained/BestHiCModule.pt",map_location = t.device('cpu'))
# fn = "./HiCdata/Rao2014-K562-MboI-allreps-filtered.10kb.cool"
# chrs_list = ['2' ,'4' ,'6' ,'8' ,'10' ,'12','16','17' ,'18','20','21']
# cell_line_name = "K562"

model.eval()

def Combine(d_size,jump,lens,hic_m):

    hic_m = hic_m.to(device) #### by HiHiC

    Hrmat = t.zeros_like(hic_m,dtype=t.float32)
    last_col_start = -1
    last_col_end = -1
    current_col_start = 0
    current_col_end = 0
    for l in range(0,lens,jump):

        lifb = False
        current_col_start = l
        current_col_end = l + d_size
        if(l + d_size >= lens):
            l = lens - d_size
            current_col_start = l
            current_col_end = l + d_size
            lifb = True

        current_col_idx = np.arange(current_col_start,current_col_end)
        last_col_idx = np.arange(last_col_start,last_col_end)
        hr_mat_col_site = np.intersect1d(current_col_idx,last_col_idx)
        small_mat_col_site = np.arange(hr_mat_col_site.shape[0])

        last_row_start = -1
        last_row_end = -1
        current_row_start = 0
        current_row_end = 0
        for c in range(l,lens,jump):
            cifb = False
            if(c + d_size >= lens):
                current_row_start = lens - d_size
                current_row_end = lens
                temp_m = hic_m[current_col_start:current_col_end,
                            current_row_start:current_row_end]
                cifb = True
            else:
                current_row_start = c
                current_row_end = c + d_size
                temp_m = hic_m[current_col_start:current_col_end,
                            current_row_start:current_row_end]
            
            result = t.triu(temp_m,diagonal=1).T + t.triu(temp_m)

            with t.no_grad():
                y = model(result.unsqueeze(0).unsqueeze(0))
            y = y.squeeze()
            enRes = t.triu(y)
            enRes = enRes.float()
            
            current_row_idx = np.arange(current_row_start,current_row_end)
            last_row_idx = np.arange(last_row_start,last_row_end)
            hr_mat_row_site = np.intersect1d(current_row_idx,last_row_idx)
            small_mat_row_site = np.arange(hr_mat_row_site.shape[0])

            if(last_row_start < 0 and last_row_end < 0 and last_col_end < 0 and last_col_start < 0):
                Hrmat[current_col_start:current_col_end,current_row_start:current_row_end] = enRes

            elif(last_col_start < 0 and last_col_end < 0 and last_row_start >= 0 and last_row_end >= 0):
                hrsub = Hrmat[:d_size,hr_mat_row_site]
                ersub = enRes[:d_size,small_mat_row_site]
                ersub[ersub == 0] = hrsub[ersub == 0]
                enRes[:d_size,small_mat_row_site] = (hrsub + ersub)/2
                Hrmat[current_col_start:current_col_end,

                    current_row_start:current_row_end] = enRes

            elif(last_row_start < 0 and last_row_end < 0 and last_col_start >= 0 and last_col_end >= 0):
                hrsub = Hrmat[hr_mat_col_site,
                            hr_mat_col_site[0]:hr_mat_col_site[0]+d_size]
                ersub = enRes[small_mat_col_site,:d_size]
                hrsub[hrsub == 0] = ersub[hrsub == 0]
                enRes[small_mat_col_site,:d_size] = (hrsub + ersub)/2
                Hrmat[current_col_start:current_col_end,
                    current_row_start:current_row_end] = enRes

            else:
                hrsub = Hrmat[hr_mat_col_site,:]
                hrsub = hrsub[:,hr_mat_row_site]
                ersub = enRes[small_mat_col_site,:]
                ersub = ersub[:,small_mat_row_site]
                ersub[ersub == 0] = hrsub[ersub == 0]
                mean_mat = (hrsub + ersub)/2 # 120 120
                enRes[:small_mat_col_site[-1]+1,
                    :small_mat_row_site[-1]+1] = mean_mat

                Hrmat[current_col_start:current_col_end,
                    current_row_start:current_row_end] = enRes

            last_row_start = current_row_start
            last_row_end = current_row_end
            if(cifb):
                break
        
        # print("\n\n\n")
        last_col_start = current_col_start
        last_col_end = current_col_end
        if(lifb): break

    Hrmat = t.triu(Hrmat,diagonal=1).T + t.triu(Hrmat)
    return Hrmat

# def Readcooler(fn,chr,b = False):
#     # print('--')
#     rdata = cooler.Cooler(fn)
    
#     rmat = rdata.matrix(balance=b).fetch(chr)
#     # rmat, _ = remove_zeros(rmat)
#     rmat[np.isnan(rmat)] = 0
#     return rmat


def predict(c):
    # rdata = Readcooler(fn,'chr' + c)
    rdata = input_data[str(c)] ###### by HiHiC
    lrmat = rdata.astype(np.float32)
    hic_m = t.from_numpy(lrmat)
    fakemat = Combine(150,50,lrmat.shape[0],hic_m)

    # np.savez('./' + cell_line_name +'HiC-Predict-chr'+c+'.npz',fakeh = fakemat.numpy(),lhr = lrmat)
    return fakemat

if __name__ == '__main__':
    # pool_num = len(chrs_list) if multiprocessing.cpu_count() > len(chrs_list) else multiprocessing.cpu_count()

    start = time.time()
    # print(f'Start a multiprocess pool with process_num = {pool_num}')
    # pool = multiprocessing.Pool(pool_num)
    # pool.apply_async(func = predict,args=(chr,))
    
    ######################### by HiHiC ####
    th_model = args.ckpt_file.split('/')[-1].split('_')[0]
    file = os.path.join(args.output_data_dir, f'{prefix}_{args.model}_{th_model}ep.npz')
    chr_dict = {}
    for chr in chrs_list:
        chr_mat = predict(chr)
        chr_dict[chr] = chr_mat.cpu().numpy()
 
    np.savez(file, **chr_dict)
    #######################################
    # pool.close()
    # pool.join()
    print(f'All predicting processes done. Running cost is {(time.time()-start)/62:.1f} min.')
