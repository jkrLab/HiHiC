import numpy as np
# import pandas as pd
import time
# import cooler
import multiprocessing


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
required.add_argument('-m', '--model', dest='model', type=str, required=True, choices=['HiCARN', 'DeepHiC', 'HiCNN', 'HiCSR', 'DFHiC', 'hicplus', 'SRHiC', 'iEnhance'])
required.add_argument('-d', '--input_downsample_dir', dest='input_downsample_dir',type=str, required=True, help='Downsampled input data directory: /HiHiC/data_downsampled_16')
required.add_argument('-g', '--ref_chrom', dest='ref_chrom', type=str, required=True, help='Reference chromosome length: /HiHiC/hg19.txt')
required.add_argument('-r', '--down_ratio', dest='down_ratio', type=str, required=True, help='Downsampling ratio: 16')
required.add_argument('-o', '--output_dir', dest='output_dir', type=str, required=True, help='Parent directory of output: /data/HiHiC/')
required.add_argument('-s', '--max_value', dest='max_value', type=str, required=True, default='300', help='Maximum value of chromosome matrix')
required.add_argument('-n', '--normalization', dest='normalization', type=str, required=True, default='None', help='Normalization method')
optional.add_argument('-t', '--train_set', dest='train_set', type=str, required=False, help='Train set chromosome: "1 2 3 4 5 6 7 8 9 10 11 12 13 14"')
optional.add_argument('-v', '--valid_set', dest='valid_set', type=str, required=False, help='Validation set chromosome: "15 16 17"')
optional.add_argument('-p', '--test_set', dest='test_set', type=str, required=False, help='Prediction set chromosome: "18 19 20 21 22"')

args = parser.parse_args()
chrs_list = args.train_set.split() + args.valid_set.split() + args.test_set.split()
input_data_dir = args.input_data_dir
input_downsample_dir = args.input_downsample_dir
ref_chrom = args.ref_chrom
ratio = args.down_ratio
output_dir = args.output_dir

normalization = args.normalization
max_value = args.max_value # minmax scaling
save_dir = f'{args.output_dir}/data_{args.model}/chrs_{normalization}_{max_value}/'
os.makedirs(save_dir, exist_ok=True)

def hic_matrix_extraction(chrs_list, res=10000):
    hr_contacts_dict={}
    for each in chrs_list:
        hr_hic_file = f'{input_data_dir}/chr{each}_10kb.txt'
        chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open(f'{ref_chrom}').readlines()} # GM12878 Hg19
        mat_dim = int(math.ceil(chrom_len[f'chr{each}']*1.0/res))
        hr_contact_matrix = np.zeros((mat_dim,mat_dim))
        for line in open(hr_hic_file).readlines():
            idx1, idx2, value = int(line.strip().split('\t')[0]),int(line.strip().split('\t')[1]),float(line.strip().split('\t')[2])
            if idx2/res>=mat_dim or idx1/res>=mat_dim:
                continue
            else:
                hr_contact_matrix[int(idx1/res)][int(idx2/res)] = value
        hr_contact_matrix+= hr_contact_matrix.T - np.diag(hr_contact_matrix.diagonal())
        hr_contacts_dict[f'chr{each}'] = scaler.fit_transform(np.minimum(int(max_value), hr_contact_matrix)) # (0,300) >> (0,1)

    lr_contacts_dict={}
    for each in chrs_list:
        lr_hic_file = f'{input_downsample_dir}/chr{each}_10kb.txt'
        chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open(f'{ref_chrom}').readlines()}
        mat_dim = int(math.ceil(chrom_len[f'chr{each}']*1.0/res))
        lr_contact_matrix = np.zeros((mat_dim,mat_dim))
        for line in open(lr_hic_file).readlines():
            idx1, idx2, value = int(line.strip().split('\t')[0]),int(line.strip().split('\t')[1]),float(line.strip().split('\t')[2])
            if idx2/res>=mat_dim or idx1/res>=mat_dim:
                continue
            else:
                lr_contact_matrix[int(idx1/res)][int(idx2/res)] = value
        lr_contact_matrix+= lr_contact_matrix.T - np.diag(lr_contact_matrix.diagonal())
        lr_contacts_dict[f'chr{each}'] = scaler.fit_transform(np.minimum(int(max_value), lr_contact_matrix)) # (0,300) >> (0,1)

    ct_hr_contacts={item:sum(sum(hr_contacts_dict[item])) for item in hr_contacts_dict.keys()} # read 수
    ct_lr_contacts={item:sum(sum(lr_contacts_dict[item])) for item in lr_contacts_dict.keys()}    
    print("  ...Done making whole contact map by each chromosomes...\n", flush=True)
    return hr_contacts_dict,lr_contacts_dict,ct_hr_contacts,ct_lr_contacts

#########################################################



# fn = "./data/Rao2014-GM12878-MboI-allreps-filtered.10kb.cool"
scale = 8
# out_path = "./divide-path/"

bssum = [600,4000,600]
## log_name = "log-divide.txt"

def remove_zeros(matrix):
    idxy = ~np.all(np.isnan(matrix), axis=0)
    M = matrix[idxy, :]
    M = M[:, idxy]
    M = np.asarray(M)
    idxy = np.asarray(idxy)
    return M, idxy

def DownSampling(rmat,ratio = 2):
    sampling_ratio = ratio
    m = np.matrix(rmat)

    all_sum = m.sum(dtype='float') # 합계 scalar 값 (전체 reads 수)
    m = m.astype(np.float64)
    idx_prob = np.divide(m, all_sum,out=np.zeros_like(m), where=all_sum != 0) # 전체 합으로 나누기 (차원 동일) 
    idx_prob = np.asarray(idx_prob.reshape(
        (idx_prob.shape[0]*idx_prob.shape[1],))) # (m, n) >>> (m*n,) 으로 flatten
    idx_prob = np.squeeze(idx_prob) # (m*n)

    sample_number_counts = int(all_sum/(2*sampling_ratio)) # 전체 reads 수 / sampling ratio*2
    # 0 1 2 ... 8
    id_range = np.arange(m.shape[0]*m.shape[1]) # (m*n)
    id_x = np.random.choice(
        id_range, size=sample_number_counts, replace=True, p=idx_prob) # m*n개의 idx 중에 size 만큼 랜덤 중복 선택, 각 요소가 선택될 확률은 idx_prob : sample_number_counts 만큼의 index

    sample_m = np.zeros_like(m)
    for i in np.arange(sample_number_counts): # 정한 reads 만큼 for loop
        x = int(id_x[i]/m.shape[0]) # index를 m으로 나눈 몫 : x 좌표
        y = int(id_x[i] % m.shape[0]) # index를 m으로 나눈 나머지 : y 좌표
        sample_m[x, y] += 1.0
    sample_m = np.transpose(sample_m) + sample_m # symmetric 하게 만들어 주기 위한 과정 (그래서 앞에서 sampling ratio 의 2배로 나누었던 것 같다)
    print("after sample :",sample_m.sum(), flush=True) # 선택된 read (== sample_number_counts)

    return np.array(sample_m) # (m, m)

def divide_hicm(hic_m,d_size,jump):
    '''
    
    '''
    lens = hic_m.shape[0] # 6
    out = [] # np.zeros(d_size ** 2).reshape(d_size,d_size)
    all_sum = 0

    for l in range(0,lens,jump):
        lifb = False
        if(l + d_size >= lens):
            l = lens - d_size
            lifb = True

        for c in range(l,lens,jump):
            cifb = False
            if(c + d_size >= lens):
                temp_m = hic_m[l:l+d_size,lens - d_size:lens]
                cifb = True
            else:
                temp_m = hic_m[l:l+d_size,c:c+d_size]
            result = np.triu(temp_m,k=1).T + np.triu(temp_m)
            all_sum += 1
            out.append(result)
            if(cifb):
                break
        if(lifb): break
    return all_sum,np.array(out)
def wlog(fname,w):
    with open(fname,'a+') as f:
        f.write(w)
        f.close()
# chrX is optional!
## chrs_list = ['1' ,'2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8' ,'9' ,'10' ,'11' ,'12' 
##              ,'13' ,'14' ,'15' ,'16' ,'17' ,'18' ,'19' ,'20' ,'21' ,'22']


# def divide(c):
def divide(c, rmat, lrmat, save_dir):
    print(f"Processing chromosome {c}...", flush=True)
    # start = time.time()
    # Step1 ReadMat
    ## rdata = cooler.Cooler(fn)
    # balance = T in part KRnorm!!! 
    ## rmat = rdata.matrix(balance=False).fetch('chr' + c)
    # rmat, _ = remove_zeros(rmat)

    # Step2 Downsampling and Norm
    # logw = 'chr ' + c + " rmat sum :" + str(rmat.sum())
    # lrmat = DownSampling(rmat,scale ** 2)
    # logw = logw + "\n" + 'chr ' + c + " lrmat sum:" + str(lrmat.sum()) + "\n"
    ## wlog(log_name,logw)

    # Step3 divide
    hrn,piece_hr = divide_hicm(rmat,150,30)
    lrn,piece_lr = divide_hicm(lrmat,150,30)

    # Step5 Sampling
    block_sum = piece_hr.sum(axis=2).sum(axis = 1)
    bgood = np.percentile(block_sum,80)
    bmedian = np.percentile(block_sum,60)

    layeridx = [np.array(block_sum <= bmedian),
                np.array(block_sum >= bgood),
                np.array(block_sum > bmedian) & np.array(block_sum < bgood)]

    hrout = []
    lrout = []

    for i in range(len(layeridx)):
        tempi = layeridx[i].sum()
        if(i == 0):
            if(tempi <= bssum[i]):
                lrout.append(piece_lr[layeridx[i]])
                hrout.append(piece_hr[layeridx[i]])
            else:
                ridx = np.random.choice(np.arange(tempi),bssum[i],replace=False)
                lrout.append(piece_lr[layeridx[i]][ridx])
                hrout.append(piece_hr[layeridx[i]][ridx])
        elif(i == 1):
            if(tempi <= bssum[i]):
                lrout.append(piece_lr[layeridx[i]])
                hrout.append(piece_hr[layeridx[i]])
            else:
                ridx = np.random.choice(np.arange(tempi),bssum[i],replace=False)
                lrout.append(piece_lr[layeridx[i]][ridx])
                hrout.append(piece_hr[layeridx[i]][ridx])
        else:
            if(tempi <= bssum[i]):
                lrout.append(piece_lr[layeridx[i]])
                hrout.append(piece_hr[layeridx[i]])
            else:
                ridx = np.random.choice(np.arange(tempi),bssum[i],replace=False)
                lrout.append(piece_lr[layeridx[i]][ridx])
                hrout.append(piece_hr[layeridx[i]][ridx])

    # Step6 Save
    piece_hr = np.concatenate(hrout,axis=0)
    piece_lr = np.concatenate(lrout,axis=0)
    # np.savez(out_path + "chr-" + c + ".npz",hr_sample = piece_hr,lr_sample = piece_lr)

    np.savez(save_dir + c ,hr_sample = piece_hr,lr_sample = piece_lr)
    ## logw = 'chr ' + c + " hr sample sum :" + str(piece_hr.shape)
    ## logw = logw + "\n" + 'chr ' + c + " lr sample sum:" + str(piece_lr.shape) + "\n"
    ## wlog(log_name,logw)

    # Step7 time
    ## cost_time = time.time() - start
    ## wlog(log_name,"time" + str(cost_time) + "\n\n")

if __name__ == '__main__':
    # pool_num = len(chrs_list) if multiprocessing.cpu_count() > len(chrs_list) else multiprocessing.cpu_count()

    # start = time.time()
    # print(f'Start a multiprocess pool with process_num = {pool_num}')
    # pool = multiprocessing.Pool(pool_num)
    
    ################################## by HiHiC #####
    hr_contacts_dict,lr_contacts_dict,_,_ = hic_matrix_extraction(chrs_list)
    chrs_list = hr_contacts_dict.keys()
    #################################################
    
    for chr in chrs_list:
        # pool.apply_async(func = divide, args=(chr,))
        rmat, lmat = hr_contacts_dict[chr],lr_contacts_dict[chr]
    #     pool.apply(func = divide, args=(chr,rmat,lmat,save_dir))
        divide(chr,rmat,lmat,save_dir)
    # pool.close()
    # pool.join()
    print(f'All downsampling processes done. Running cost is {(time.time()-start)/60:.1f} min.', flush=True)
    print("All is done ! ^_^ ", flush=True)