# DFHiC/generate_train_data.py 수정

import os, sys, math, random, argparse
import numpy as np

path = os.getcwd()
random.seed(100)


# 인자 받기
parser = argparse.ArgumentParser(description='Read Hi-C contact map and Divide submatrix for train and predict', add_help=True)
req_args = parser.add_argument_group('Required Arguments')
req_args.add_argument('-i', dest='input_data_dir', required=True,
                      help='REQUIRED: Hi-C data directory containing .txt files (Directory of Hi-C contact pares) ==== (example) /HiHiC-main/data ====')
req_args.add_argument('-d', dest='input_downsample_dir', required=True,
                      help='REQUIRED: Hi-C downsampled data directory containing .txt files (Directory of downsampled Hi-C contact pares) ==== (example) /HiHiC-main/data_downsampled_16 ====')
req_args.add_argument('-m', dest='model', required=True, choices=['HiCARN', 'DeepHiC', 'HiCNN2', 'HiCSR', 'DFHiC', 'hicplus', 'SRHiC'],
                      help='REQUIRED: Model name that you want to use (One of HiCARN, DeepHiC, HiCNN2, HiCSR, DFHiC, hicplus, and SRHiC) ==== (example) DFHiC ====')
req_args.add_argument('-g', dest='ref_chrom', required=True,
                      help='REQUIRED: Reference genome length file, your data is based on ==== (example) /HiHiC-main/hg19.txt ====')
req_args.add_argument('-r', dest='data_ratio', required=True,
                      help='REQUIRED: Downsampling ratio of your downsampled data ==== (example) 16 ====')
req_args.add_argument('-o', dest='output_dir', required=True,
                      help='REQUIRED: Parent directory path for saving output (Child directory named as the model name will be generated under this.) ==== (example) /HiHiC ====')

args = parser.parse_args()
input_data_dir = args.input_data_dir 
input_downsample_dir = args.input_downsample_dir
model = args.model
ref_chrom = args.ref_chrom
data_ratio = args.data_ratio
output_dir = args.output_dir
print(f"\n  ...Start generating input data for {model} training...\n    using Hi-C data of {input_data_dir} and 1/{data_ratio} downsampled data of {input_downsample_dir}")


# 크로모좀 별로 matrix 만들기
def hic_matrix_extraction(res=10000, norm_method='NONE'):
    chrom_list = list(range(1,23))#chr1-chr22
    hr_contacts_dict={}
    for each in chrom_list:
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
        hr_contacts_dict[f'chr{each}'] = hr_contact_matrix
    lr_contacts_dict={}
    for each in chrom_list:
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
        lr_contacts_dict[f'chr{each}'] = lr_contact_matrix

    nb_hr_contacts={item:sum(sum(hr_contacts_dict[item])) for item in hr_contacts_dict.keys()} # read 수
    nb_lr_contacts={item:sum(sum(lr_contacts_dict[item])) for item in lr_contacts_dict.keys()}
    print("\n  ...Done making whole contact map by each chromosomes...")
     
    return hr_contacts_dict,lr_contacts_dict,nb_hr_contacts,nb_lr_contacts


# 매트릭스 자르기
def crop_hic_matrix_by_chrom(chrom, size, for_model, thred=200): # thred=2M/resolution
    chr = int(chrom.split('chr')[1])
    distance=[]
    crop_mats_hr=[]
    crop_mats_lr=[]
    coordinates_hr=[]    
    coordinates_lr=[]    

    row,col = hr_contacts_dict[chrom].shape
    if row<=thred or col<=thred: # bin 수가 200 보다 작으면 False
        print('HiC matrix size wrong!')
        sys.exit()
    def quality_control(mat,thred=0.05):
        if len(mat.nonzero()[0])<thred*mat.shape[0]*mat.shape[1]: # 숫자 있는 셀의 수가 전체의 5% 미만이면 False
            return False
        else:
            return True
    
    if size == 40:
        if for_model == "HiCNN":
            for idx1 in range(0,row-size,28):
                for idx2 in range(0,col-size,28):
                    if abs(idx1-idx2)<thred:
                        if quality_control(lr_contacts_dict[chrom][idx1:idx1+size,idx2:idx2+size]):
                            distance.append([idx1-idx2,chrom])
                            coordinates_hr.append([chr, idx1, idx2])
                            coordinates_lr.append([chr, idx1, idx2])
                            
                            hr_contact = hr_contacts_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                            lr_contact = lr_contacts_dict[chrom][idx1:idx1+size,idx2:idx2+size]

                            crop_mats_hr.append(hr_contact)                
                            crop_mats_lr.append(lr_contact)

            crop_mats_hr = np.concatenate([item[np.newaxis,:] for item in crop_mats_hr],axis=0)
            crop_mats_lr = np.concatenate([item[np.newaxis,:] for item in crop_mats_lr],axis=0)

        else:        
            for idx1 in range(0,row-size,size):
                for idx2 in range(0,col-size,size):
                    if abs(idx1-idx2)<thred:
                        if quality_control(lr_contacts_dict[chrom][idx1:idx1+size,idx2:idx2+size]):
                            distance.append([idx1-idx2,chrom])
                            coordinates_hr.append([chr, idx1, idx2])
                            coordinates_lr.append([chr, idx1, idx2])
                            
                            hr_contact = hr_contacts_dict[chrom][idx1:idx1+size,idx2:idx2+size]
                            lr_contact = lr_contacts_dict[chrom][idx1:idx1+size,idx2:idx2+size]

                            crop_mats_hr.append(hr_contact)                
                            crop_mats_lr.append(lr_contact)

            crop_mats_hr = np.concatenate([item[np.newaxis,:] for item in crop_mats_hr],axis=0)
            crop_mats_lr = np.concatenate([item[np.newaxis,:] for item in crop_mats_lr],axis=0)                
    else:
        assert size == 28
        for idx1 in range(0,row-40,size):
            for idx2 in range(0,col-40,size):
                if abs(idx1-idx2)<thred:
                    if quality_control(lr_contacts_dict[chrom][idx1:idx1+size,idx2:idx2+size]):
                        distance.append([idx1-idx2,chrom])
                        coordinates_hr.append([chr, idx1+6, idx2+6])
                        coordinates_lr.append([chr, idx1, idx2])
                        
                        hr_contact = hr_contacts_dict[chrom][idx1+6:idx1+34,idx2+6:idx2+34]
                        lr_contact = lr_contacts_dict[chrom][idx1:idx1+40,idx2:idx2+40]

                        crop_mats_hr.append(hr_contact)                
                        crop_mats_lr.append(lr_contact)
                      
        crop_mats_hr = np.concatenate([item[np.newaxis,:] for item in crop_mats_hr],axis=0)
        crop_mats_lr = np.concatenate([item[np.newaxis,:] for item in crop_mats_lr],axis=0)
        
    return crop_mats_hr,crop_mats_lr,distance,coordinates_hr,coordinates_lr


# 모델별 submatrix 생성
def DeepHiC_data_split(chrom_list):
    distance_all=[]
    assert len(chrom_list)>0
    hr_mats,lr_mats,hr_coordinates=[],[],[]
    for chrom in chrom_list:
        crop_mats_hr,crop_mats_lr,distance,coordinates_hr,_ = crop_hic_matrix_by_chrom(chrom,size=40,for_model='DeepHiC',thred=200)
        distance_all+=distance
        hr_mats.append(crop_mats_hr)
        lr_mats.append(crop_mats_lr)
        hr_coordinates.append(coordinates_hr)
    hr_mats = np.concatenate(hr_mats,axis=0)
    lr_mats = np.concatenate(lr_mats,axis=0)
    hr_mats=hr_mats[:,np.newaxis]
    lr_mats=lr_mats[:,np.newaxis]
    hr_coordinates = sum(hr_coordinates, [])
    return hr_mats,lr_mats,hr_coordinates

def HiCARN_data_split(chrom_list):
    distance_all=[]
    assert len(chrom_list)>0
    hr_mats,lr_mats,hr_coordinates=[],[],[]
    for chrom in chrom_list:
        crop_mats_hr,crop_mats_lr,distance,coordinates_hr,_ = crop_hic_matrix_by_chrom(chrom,size=40,for_model='HiCARN',thred=200)
        distance_all+=distance
        hr_mats.append(crop_mats_hr)
        lr_mats.append(crop_mats_lr)
        hr_coordinates.append(coordinates_hr)
    hr_mats = np.concatenate(hr_mats,axis=0)
    lr_mats = np.concatenate(lr_mats,axis=0)
    hr_mats=hr_mats[:,np.newaxis]
    lr_mats=lr_mats[:,np.newaxis]
    hr_coordinates = sum(hr_coordinates, [])
    return hr_mats,lr_mats,hr_coordinates

def DFHiC_data_split(chrom_list):
    distance_all=[]
    assert len(chrom_list)>0
    hr_mats,lr_mats=[],[]
    for chrom in chrom_list:
        crop_mats_hr,crop_mats_lr,distance,_,_ = crop_hic_matrix_by_chrom(chrom,size=40,for_model='DFHiC',thred=200)
        distance_all+=distance
        hr_mats.append(crop_mats_hr)
        lr_mats.append(crop_mats_lr)
    hr_mats = np.concatenate(hr_mats,axis=0)
    lr_mats = np.concatenate(lr_mats,axis=0)
    hr_mats=hr_mats[:,np.newaxis]
    lr_mats=lr_mats[:,np.newaxis]
    hr_mats=hr_mats.transpose((0,2,3,1))
    lr_mats=lr_mats.transpose((0,2,3,1))
    return hr_mats,lr_mats,distance_all

def HiCNN_data_split(chrom_list):
    distance_all=[]
    assert len(chrom_list)>0
    hr_mats,lr_mats,hr_coordinates,lr_coordinates=[],[],[],[]
    for chrom in chrom_list:
        crop_mats_hr,crop_mats_lr,distance,coordinates_hr,coordinates_lr = crop_hic_matrix_by_chrom(chrom,size=40,for_model='HiCNN',thred=200)
        distance_all+=distance
        hr_mats.append(crop_mats_hr)
        lr_mats.append(crop_mats_lr)
        hr_coordinates.append(coordinates_hr)
        lr_coordinates.append(coordinates_lr)
    hr_mats = np.concatenate(hr_mats,axis=0)
    lr_mats = np.concatenate(lr_mats,axis=0)
    hr_mats=hr_mats[:,np.newaxis]
    lr_mats=lr_mats[:,np.newaxis]
    hr_coordinates = sum(hr_coordinates, [])
    lr_coordinates = sum(lr_coordinates, [])
    return hr_mats,lr_mats,hr_coordinates,lr_coordinates

def SRHiC_data_split(chrom_list):
    distance_all=[]
    assert len(chrom_list)>0
    hr_mats,lr_mats=[],[]
    for chrom in chrom_list:
        crop_mats_hr,crop_mats_lr,_,_,_ = crop_hic_matrix_by_chrom(chrom,size=28,for_model='SRHiC',thred=200)
        hr_mats.append(crop_mats_hr)
        lr_mats.append(crop_mats_lr)
    hr_mats = np.concatenate(hr_mats,axis=0)
    lr_mats = np.concatenate(lr_mats,axis=0)
    hr_mats=hr_mats[:,np.newaxis]
    lr_mats=lr_mats[:,np.newaxis]
    return hr_mats,lr_mats

def hicplus_data_split(chrom_list):
    distance_all=[]
    assert len(chrom_list)>0
    hr_mats,lr_mats,hr_coordinates,lr_coordinates=[],[],[],[]
    for chrom in chrom_list:
        crop_mats_hr,crop_mats_lr,distance,coordinates_hr,coordinates_lr = crop_hic_matrix_by_chrom(chrom,size=28,for_model='hicplus',thred=200)
        distance_all+=distance
        hr_mats.append(crop_mats_hr)
        lr_mats.append(crop_mats_lr)
        hr_coordinates.append(coordinates_hr)
        lr_coordinates.append(coordinates_lr)
    hr_mats = np.concatenate(hr_mats,axis=0)
    lr_mats = np.concatenate(lr_mats,axis=0)
    hr_mats=hr_mats[:,np.newaxis]
    lr_mats=lr_mats[:,np.newaxis]
    hr_coordinates = sum(hr_coordinates, [])
    lr_coordinates = sum(lr_coordinates, [])
    return hr_mats,lr_mats,hr_coordinates,lr_coordinates


# 함수 실행
hr_contacts_dict,lr_contacts_dict,nb_hr_contacts,nb_lr_contacts = hic_matrix_extraction()

save_dir = f'{output_dir}data_{model}/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)   

# 모델이 원하는 포멧으로 저장
if model == "DFHiC":
    hr_mats_train,lr_mats_train,distance_train = DFHiC_data_split([f'chr{idx}' for idx in list(range(1,18))]) # train: 1~17
    hr_mats_test,lr_mats_test,distance_test = DFHiC_data_split([f'chr{idx}' for idx in list(range(18,23))]) # test: 18~22
    print(f"\n  ...Done cropping whole matrix into submatrix for {model} training...")

    np.savez(save_dir+f'train_data_raw_ratio{data_ratio}.npz', train_lr=lr_mats_train,train_hr=hr_mats_train,distance=distance_train)
    np.savez(save_dir+f'test_data_raw_ratio{data_ratio}.npz', test_lr=lr_mats_test,test_hr=hr_mats_test,distance=distance_test)

elif model == "deepHiC":      
    hr_mats_train,lr_mats_train,coordinates_train = DeepHiC_data_split([f'chr{idx}' for idx in list(range(1,15))]) # train:1~15
    hr_mats_valid,lr_mats_valid,coordinates_valid = DeepHiC_data_split([f'chr{idx}' for idx in list(range(15,18))]) # valid:15~17
    hr_mats_test,lr_mats_test,coordinates_test = DeepHiC_data_split([f'chr{idx}' for idx in list(range(18,23))]) # test:18~22
    print(f"\n  ...Done cropping whole matrix into submatrix for {model} training...")

    compacts = {int(k.split('chr')[1]) : np.nonzero(v)[0] for k, v in hr_contacts_dict.items()}
    size = {item.split()[0].split('chr')[1]:int(item.strip().split()[1])for item in open(f'{ref_chrom}').readlines()}

    os.mkdir(save_dir+'Train_and_Validation/')
    os.mkdir(save_dir+'Test/')

    np.savez(save_dir+f'Train_and_Validation/train_ratio{data_ratio}.npz', data=lr_mats_train,target=hr_mats_train,inds=np.array(coordinates_train, dtype=np.int_),compacts=compacts,size=size)
    np.savez(save_dir+f'Train_and_Validation/valid_ratio{data_ratio}.npz', data=lr_mats_valid,target=hr_mats_valid,inds=np.array(coordinates_valid, dtype=np.int_),compacts=compacts,size=size)
    np.savez(save_dir+f'Test/test_ratio{data_ratio}.npz', data=lr_mats_test,target=hr_mats_test,inds=np.array(coordinates_test, dtype=np.int_),compacts=compacts,size=size)
    
elif model == "HiCARN":          
    hr_mats_train,lr_mats_train,coordinates_train = HiCARN_data_split([f'chr{idx}' for idx in list(range(1,15))]) # train:1~14
    hr_mats_valid,lr_mats_valid,coordinates_valid = HiCARN_data_split([f'chr{idx}' for idx in list(range(15,18))]) # valid:15~17
    hr_mats_test,lr_mats_test,coordinates_test = HiCARN_data_split([f'chr{idx}' for idx in list(range(18,23))]) # test:18~22
    print(f"\n  ...Done cropping whole matrix into submatrix for {model} training...")

    compacts = {int(k.split('chr')[1]) : np.nonzero(v)[0] for k, v in hr_contacts_dict.items()}
    size = {item.split()[0].split('chr')[1]:int(item.strip().split()[1])for item in open(f'{ref_chrom}').readlines()}

    os.mkdir(save_dir+'Train_and_Validation/')
    os.mkdir(save_dir+'Test/')

    np.savez(save_dir+f'Train_and_Validation/train_ratio{data_ratio}.npz', data=lr_mats_train,target=hr_mats_train,inds=np.array(coordinates_train, dtype=np.int_),compacts=compacts,size=size)
    np.savez(save_dir+f'Train_and_Validation/valid_ratio{data_ratio}.npz', data=lr_mats_valid,target=hr_mats_valid,inds=np.array(coordinates_valid, dtype=np.int_),compacts=compacts,size=size)
    np.savez(save_dir+f'Test/test_ratio{data_ratio}.npz', data=lr_mats_test,target=hr_mats_test,inds=np.array(coordinates_test, dtype=np.int_),compacts=compacts,size=size)

elif model == "HiCNN" or "HiCNN2":     
    hr_mats_train,lr_mats_train,hr_coordinates_train,lr_coordinates_train = HiCNN_data_split([f'chr{idx}' for idx in list(range(1,15))]) # train:1~14
    hr_mats_valid,lr_mats_valid,hr_coordinates_valid,lr_coordinates_valid = HiCNN_data_split([f'chr{idx}' for idx in list(range(15,18))]) # valid:15~17
    # hr_mats_test,lr_mats_test,hr_coordinates_test,lr_coordinates_test = HiCNN_data_split([f'chr{idx}' for idx in list(range(18,23))]) # test:18~22
    print(f"\n  ...Done cropping whole matrix into submatrix for {model} training...")

    np.save(save_dir+f'subMats_train_target_ratio{data_ratio}', hr_mats_train)
    np.save(save_dir+f'subMats_train_ratio{data_ratio}', lr_mats_train)
    np.save(save_dir+f'index_train_target', hr_coordinates_train)
    np.save(save_dir+f'index_train_data', lr_coordinates_train)
    np.save(save_dir+f'subMats_valid_target_ratio{data_ratio}', hr_mats_valid)
    np.save(save_dir+f'subMats_valid_ratio{data_ratio}', lr_mats_valid)
    np.save(save_dir+f'index_valid_target', hr_coordinates_valid)
    np.save(save_dir+f'index_valid_data', lr_coordinates_valid)
    
elif model == "SRHiC":  
    hr_mats_train,lr_mats_train = SRHiC_data_split([f'chr{idx}' for idx in list(range(1,18))]) # train: 1~17
    hr_mats_test,lr_mats_test = SRHiC_data_split([f'chr{idx}' for idx in list(range(18,23))]) # valid:15~17
    print(f"\n  ...Done cropping whole matrix into submatrix for {model} training...")

    train = np.concatenate((lr_mats_train[:,0,:,:], np.concatenate((hr_mats_train[:,0,:,:],np.zeros((hr_mats_train.shape[0],12,28))), axis=1)), axis=2)
    valid = np.concatenate((lr_mats_test[:,0,:,:], np.concatenate((hr_mats_test[:,0,:,:],np.zeros((hr_mats_test.shape[0],12,28))), axis=1)), axis=2)

    np.save(save_dir+f'train_data_raw_ratio{data_ratio}', train)
    np.save(save_dir+f'valid_data_raw_ratio{data_ratio}', valid)
    
else:
    assert model == "hicplus"
    hr_mats_train,lr_mats_train,hr_coordinates_train,lr_coordinates_train = hicplus_data_split([f'chr{idx}' for idx in list(range(1,18))]) # train:1~17
    hr_mats_test,lr_mats_test,hr_coordinates_test,lr_coordinates_test = hicplus_data_split([f'chr{idx}' for idx in list(range(18,23))]) # test:18~22
    print(f"\n  ...Done cropping whole matrix into submatrix for {model} training...")

    np.save(save_dir+f'subMats_train_target_ratio{data_ratio}', hr_mats_train)
    np.save(save_dir+f'subMats_train_ratio{data_ratio}', lr_mats_train)
    np.save(save_dir+f'index_train_target', hr_coordinates_train)
    np.save(save_dir+f'index_train_data', lr_coordinates_train)
    np.save(save_dir+f'subMats_test_target_ratio{data_ratio}', hr_mats_test)
    np.save(save_dir+f'subMats_test_ratio{data_ratio}', lr_mats_test)
    np.save(save_dir+f'index_test_target', hr_coordinates_test)
    np.save(save_dir+f'index_test_data', lr_coordinates_test)  
    
print(f"\n  ...Generated data is saved in {save_dir}...\n")