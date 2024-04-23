# DFHiC/generate_train_data.py 수정

import os, sys, math, random, argparse, shutil
import numpy as np
from sklearn.preprocessing import MinMaxScaler

path = os.getcwd()
random.seed(100)
scaler = MinMaxScaler(feature_range=(0,1))

# 인자 받기
parser = argparse.ArgumentParser(description='Read Hi-C contact map and Divide submatrix for train and predict', add_help=True)
req_args = parser.add_argument_group('Required Arguments')
req_args.add_argument('-i', dest='input_data_dir', required=True,
                      help='REQUIRED: Hi-C data directory containing .txt files (Directory of Hi-C contact pares) ==== (example) /HiHiC-main/data ====')
req_args.add_argument('-d', dest='input_downsample_dir', required=True,
                      help='REQUIRED: Hi-C downsampled data directory containing .txt files (Directory of downsampled Hi-C contact pares) ==== (example) /HiHiC-main/data_downsampled_16 ====')
req_args.add_argument('-m', dest='model', required=True, choices=['HiCARN', 'DeepHiC', 'HiCNN', 'HiCSR', 'DFHiC', 'hicplus', 'SRHiC'],
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
        if np.isnan(hr_contact_matrix).any(): ###############
            print(f'hr_chr{each} has nan value!') ###############
        hr_contacts_dict[f'chr{each}'] = scaler.fit_transform(np.minimum(300, hr_contact_matrix)) # (0,300) >> (0,1)
        if np.isnan(hr_contacts_dict[f'chr{each}']).any(): ###############
            print(f'hr_scaled chr{each} has nan value!') ###############

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
        if np.isnan(lr_contact_matrix).any(): ###############
            print(f'lr_chr{each} has nan value!') ###############
        lr_contacts_dict[f'chr{each}'] = scaler.fit_transform(np.minimum(300, lr_contact_matrix)) # (0,300) >> (0,1)
        if np.isnan(lr_contacts_dict[f'chr{each}']).any(): ###############
            print(f'lr_scaled chr{each} has nan value!') ###############

    ct_hr_contacts={item:sum(sum(hr_contacts_dict[item])) for item in hr_contacts_dict.keys()} # read 수
    ct_lr_contacts={item:sum(sum(lr_contacts_dict[item])) for item in lr_contacts_dict.keys()}    
    print("\n  ...Done making whole contact map by each chromosomes...")
    return hr_contacts_dict,lr_contacts_dict,ct_hr_contacts,ct_lr_contacts


# 매트릭스 자르기
def crop_hic_matrix_by_chrom(chrom, for_model, thred=200): # thred=2M/resolution
    chr = int(chrom.split('chr')[1])
    distance=[]
    hr_crop_mats=[]
    lr_crop_mats=[]
    hr_coordinates=[]    
    lr_coordinates=[]
    row,col = hr_contacts_dict[chrom].shape
    if row<=thred or col<=thred: # bin 수가 200 보다 작으면 False
        print('HiC matrix size wrong!')
        sys.exit()
    def quality_control(mat,thred=0.05):
        if len(mat.nonzero()[0])<thred*mat.shape[0]*mat.shape[1]: # 숫자 있는 셀의 수가 전체의 5% 미만이면 False
            return False
        else:
            return True
    for idx1 in range(0,row-40,28):
        for idx2 in range(0,col-40,28):
            if abs(idx1-idx2)<thred:
                if quality_control(lr_contacts_dict[chrom][idx1:idx1+40,idx2:idx2+40]):
                    distance.append([idx1-idx2,chrom]) # DFHiC
                    lr_coordinates.append([chr, row, idx1, idx2]) # row:HiCARN                    
                    lr_contact = lr_contacts_dict[chrom][idx1:idx1+40,idx2:idx2+40]
                    if for_model in ["HiCARN", "DeepHiC", "DFHiC"]:
                        hr_contact = hr_contacts_dict[chrom][idx1:idx1+40,idx2:idx2+40]
                        hr_coordinates.append([chr, row, idx1, idx2])
                    elif for_model in ["HiCNN", "hicplus"]: # output mat:28*28
                        hr_contact = hr_contacts_dict[chrom][idx1:idx1+40,idx2:idx2+40]
                        hr_coordinates.append([chr, row, idx1+6, idx2+6])
                    else:
                        assert for_model in ["SRHiC"]
                        hr_contact = hr_contacts_dict[chrom][idx1+6:idx1+34,idx2+6:idx2+34]
                        hr_coordinates.append([chr, row, idx1+6, idx2+6]) 
                    hr_crop_mats.append(hr_contact)                
                    lr_crop_mats.append(lr_contact)
    hr_crop_mats = np.concatenate([item[np.newaxis,:] for item in hr_crop_mats],axis=0)
    lr_crop_mats = np.concatenate([item[np.newaxis,:] for item in lr_crop_mats],axis=0)
    return hr_crop_mats,lr_crop_mats,hr_coordinates,lr_coordinates,distance     


# 모델별 submatrix 생성
def DeepHiC_data_split(chrom_list):
    assert len(chrom_list)>0
    hr_mats,lr_mats,hr_coords=[],[],[]
    for chrom in chrom_list:
        hr_crop_mats,lr_crop_mats,hr_coordinates,_,_ = crop_hic_matrix_by_chrom(chrom,for_model="DeepHiC",thred=200)
        hr_mats.append(hr_crop_mats)
        lr_mats.append(lr_crop_mats)
        hr_coords.append(hr_coordinates)
    hr_mats = np.concatenate(hr_mats,axis=0)
    lr_mats = np.concatenate(lr_mats,axis=0)
    hr_mats=hr_mats[:,np.newaxis]
    lr_mats=lr_mats[:,np.newaxis]
    hr_coords = sum(hr_coords, [])
    return hr_mats,lr_mats,hr_coords

def HiCARN_data_split(chrom_list):
    assert len(chrom_list)>0
    hr_mats,lr_mats,hr_coords=[],[],[]
    for chrom in chrom_list:
        hr_crop_mats,lr_crop_mats,hr_coordinates,_,_ = crop_hic_matrix_by_chrom(chrom,for_model="HiCARN",thred=200)
        hr_mats.append(hr_crop_mats)
        lr_mats.append(lr_crop_mats)
        hr_coords.append(hr_coordinates)
    hr_mats = np.concatenate(hr_mats,axis=0)
    lr_mats = np.concatenate(lr_mats,axis=0)
    hr_mats=hr_mats[:,np.newaxis]
    lr_mats=lr_mats[:,np.newaxis]
    hr_coords = sum(hr_coords, [])
    return hr_mats,lr_mats,hr_coords

def DFHiC_data_split(chrom_list):
    distance_all=[]
    assert len(chrom_list)>0
    hr_mats,lr_mats,hr_coords=[],[],[]
    for chrom in chrom_list:
        hr_crop_mats,lr_crop_mats,hr_coordinates,_,distance = crop_hic_matrix_by_chrom(chrom,for_model="DFHiC",thred=200)
        distance_all+=distance
        hr_mats.append(hr_crop_mats)
        lr_mats.append(lr_crop_mats)
        hr_coords.append(hr_coordinates)
    hr_mats = np.concatenate(hr_mats,axis=0)
    lr_mats = np.concatenate(lr_mats,axis=0)
    hr_mats=hr_mats[:,np.newaxis]
    lr_mats=lr_mats[:,np.newaxis]
    hr_mats=hr_mats.transpose((0,2,3,1))
    lr_mats=lr_mats.transpose((0,2,3,1))
    hr_coords = sum(hr_coords, [])
    return hr_mats,lr_mats,hr_coords,distance_all

def HiCNN_data_split(chrom_list):
    assert len(chrom_list)>0
    hr_mats,lr_mats,hr_coords,lr_coords=[],[],[],[]
    for chrom in chrom_list:
        hr_crop_mats,lr_crop_mats,hr_coordinates,lr_coordinates,_= crop_hic_matrix_by_chrom(chrom,for_model="HiCNN",thred=200)
        hr_mats.append(hr_crop_mats)
        lr_mats.append(lr_crop_mats)
        hr_coords.append(hr_coordinates)
        lr_coords.append(lr_coordinates)
    hr_mats = np.concatenate(hr_mats,axis=0)
    lr_mats = np.concatenate(lr_mats,axis=0)
    hr_mats=hr_mats[:,np.newaxis]
    lr_mats=lr_mats[:,np.newaxis]
    hr_coords = sum(hr_coords, [])
    lr_coords = sum(lr_coords, [])
    return hr_mats,lr_mats,hr_coords,lr_coords

def SRHiC_data_split(chrom_list):
    assert len(chrom_list)>0
    hr_mats,lr_mats,hr_coords,lr_coords=[],[],[],[]
    for chrom in chrom_list:
        hr_crop_mats,lr_crop_mats,hr_coordinates,lr_coordinates,_ = crop_hic_matrix_by_chrom(chrom,for_model="SRHiC",thred=200)
        hr_mats.append(hr_crop_mats)
        lr_mats.append(lr_crop_mats)
        hr_coords.append(hr_coordinates)
        lr_coords.append(lr_coordinates)
    hr_mats = np.concatenate(hr_mats,axis=0)
    lr_mats = np.concatenate(lr_mats,axis=0)
    hr_mats=hr_mats[:,np.newaxis]
    lr_mats=lr_mats[:,np.newaxis]
    hr_coords = sum(hr_coords, [])
    lr_coords = sum(lr_coords, [])
    return hr_mats,lr_mats,hr_coords,lr_coords

def hicplus_data_split(chrom_list):
    assert len(chrom_list)>0
    hr_mats,lr_mats,hr_coords,lr_coords=[],[],[],[]
    for chrom in chrom_list: 
        hr_crop_mats,lr_crop_mats,hr_coordinates,lr_coordinates,_ = crop_hic_matrix_by_chrom(chrom,for_model="hicplus",thred=200)
        hr_mats.append(hr_crop_mats)
        lr_mats.append(lr_crop_mats)
        hr_coords.append(hr_coordinates)
        lr_coords.append(lr_coordinates)
    hr_mats = np.concatenate(hr_mats,axis=0)
    lr_mats = np.concatenate(lr_mats,axis=0)
    hr_mats=hr_mats[:,np.newaxis]
    lr_mats=lr_mats[:,np.newaxis]
    hr_coords = sum(hr_coords, [])
    lr_coords = sum(lr_coords, [])
    return hr_mats,lr_mats,hr_coords,lr_coords


# 함수 실행
hr_contacts_dict,lr_contacts_dict,ct_hr_contacts,ct_lr_contacts = hic_matrix_extraction()

save_dir = f'{output_dir}data_{model}/'
os.makedirs(save_dir, exist_ok=True)

train_dir = save_dir + "train_KR_300/"
valid_dir = save_dir + "valid_KR_300/"
test_dir = save_dir + "test_KR_300/"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


# 모델이 원하는 포멧으로 저장
if model == "DFHiC":
    hr_mats_train,lr_mats_train,coordinates_train,distance_train = DFHiC_data_split([f'chr{idx}' for idx in list(range(1,15))]) # train: 1~14
    hr_mats_valid,lr_mats_valid,coordinates_valid,distance_valid = DFHiC_data_split([f'chr{idx}' for idx in list(range(15,18))]) # train: 15~17
    hr_mats_test,lr_mats_test,coordinates_test,distance_test = DFHiC_data_split([f'chr{idx}' for idx in list(range(18,23))]) # test: 18~22
    print(f"\n  ...Done cropping whole matrix into submatrix for {model} training...")

    np.savez(train_dir+f"train_ratio{data_ratio}.npz", data=lr_mats_train,target=hr_mats_train,inds=np.array(coordinates_train, dtype=np.int_),distance=distance_train)
    np.savez(valid_dir+f"valid_ratio{data_ratio}.npz", data=lr_mats_valid,target=hr_mats_valid,inds=np.array(coordinates_valid, dtype=np.int_),distance=distance_valid)
    np.savez(test_dir+f"test_ratio{data_ratio}.npz", data=lr_mats_test,target=hr_mats_test,inds=np.array(coordinates_test, dtype=np.int_),distance=distance_test)


elif model == "DeepHiC":      
    hr_mats_train,lr_mats_train,coordinates_train = DeepHiC_data_split([f'chr{idx}' for idx in list(range(1,15))]) # train:1~14
    hr_mats_valid,lr_mats_valid,coordinates_valid = DeepHiC_data_split([f'chr{idx}' for idx in list(range(15,18))]) # valid:15~17
    hr_mats_test,lr_mats_test,coordinates_test = DeepHiC_data_split([f'chr{idx}' for idx in list(range(18,23))]) # test:18~22
    print(f"\n  ...Done cropping whole matrix into submatrix for {model} training...")

    compacts = {int(k.split('chr')[1]) : np.nonzero(v)[0] for k, v in hr_contacts_dict.items()}
    size = {item.split()[0].split('chr')[1]:int(item.strip().split()[1])for item in open(f'{ref_chrom}').readlines()}

    np.savez(train_dir+f"train_ratio{data_ratio}.npz", data=lr_mats_train,target=hr_mats_train,inds=np.array(coordinates_train, dtype=np.int_),compacts=compacts,sizes=size)
    np.savez(valid_dir+f"valid_ratio{data_ratio}.npz", data=lr_mats_valid,target=hr_mats_valid,inds=np.array(coordinates_valid, dtype=np.int_),compacts=compacts,sizes=size)
    np.savez(test_dir+f"test_ratio{data_ratio}.npz", data=lr_mats_test,target=hr_mats_test,inds=np.array(coordinates_test, dtype=np.int_),compacts=compacts,sizes=size)
    
    
elif model == "HiCARN":          
    hr_mats_train,lr_mats_train,coordinates_train = HiCARN_data_split([f'chr{idx}' for idx in list(range(1,15))]) # train:1~14
    hr_mats_valid,lr_mats_valid,coordinates_valid = HiCARN_data_split([f'chr{idx}' for idx in list(range(15,18))]) # valid:15~17
    hr_mats_test,lr_mats_test,coordinates_test = HiCARN_data_split([f'chr{idx}' for idx in list(range(18,23))]) # test:18~22
    print(f"\n  ...Done cropping whole matrix into submatrix for {model} training...")

    compacts = {int(k.split('chr')[1]) : np.nonzero(v)[0] for k, v in hr_contacts_dict.items()}
    size = {item.split()[0].split('chr')[1]:int(item.strip().split()[1])for item in open(f'{ref_chrom}').readlines()}

    np.savez(train_dir+f"train_ratio{data_ratio}.npz", data=lr_mats_train,target=hr_mats_train,inds=np.array(coordinates_train, dtype=np.int_),compacts=compacts,sizes=size)
    np.savez(valid_dir+f"valid_ratio{data_ratio}.npz", data=lr_mats_valid,target=hr_mats_valid,inds=np.array(coordinates_valid, dtype=np.int_),compacts=compacts,sizes=size)
    np.savez(test_dir+f"test_ratio{data_ratio}.npz", data=lr_mats_test,target=hr_mats_test,inds=np.array(coordinates_test, dtype=np.int_),compacts=compacts,sizes=size)


elif model == "HiCNN":     
    hr_mats_train,lr_mats_train,hr_coordinates_train,lr_coordinates_train = HiCNN_data_split([f'chr{idx}' for idx in list(range(1,15))]) # train:1~14
    hr_mats_valid,lr_mats_valid,hr_coordinates_valid,lr_coordinates_valid = HiCNN_data_split([f'chr{idx}' for idx in list(range(15,18))]) # valid:15~17
    hr_mats_test,lr_mats_test,hr_coordinates_test,lr_coordinates_test = HiCNN_data_split([f'chr{idx}' for idx in list(range(18,23))]) # test:18~22
    print(f"\n  ...Done cropping whole matrix into submatrix for {model} training...")

    np.savez(train_dir+f"train_ratio{data_ratio}.npz", data=lr_mats_train,target=hr_mats_train,inds=np.array(lr_coordinates_train, dtype=np.int_),inds_target=np.array(hr_coordinates_train, dtype=np.int_))
    np.savez(valid_dir+f"valid_ratio{data_ratio}.npz", data=lr_mats_valid,target=hr_mats_valid,inds=np.array(lr_coordinates_valid, dtype=np.int_),inds_target=np.array(hr_coordinates_valid, dtype=np.int_))
    np.savez(test_dir+f"test_ratio{data_ratio}.npz", data=lr_mats_test,target=hr_mats_test,inds=np.array(lr_coordinates_test, dtype=np.int_),inds_target=np.array(hr_coordinates_test, dtype=np.int_))
    
    
elif model == "SRHiC":  
    hr_mats_train,lr_mats_train,hr_coordinates_train,lr_coordinates_train = SRHiC_data_split([f'chr{idx}' for idx in list(range(1,15))]) # train: 1~14
    hr_mats_valid,lr_mats_valid,hr_coordinates_valid,lr_coordinates_valid = SRHiC_data_split([f'chr{idx}' for idx in list(range(15,18))]) # valid:15~17
    hr_mats_test,lr_mats_test,hr_coordinates_test,lr_coordinates_test = SRHiC_data_split([f'chr{idx}' for idx in list(range(18,23))]) # test:18~22
    print(f"\n  ...Done cropping whole matrix into submatrix for {model} training...")

    train = np.concatenate((lr_mats_train[:,0,:,:], np.concatenate((hr_mats_train[:,0,:,:],np.zeros((hr_mats_train.shape[0],12,28))), axis=1)), axis=2)
    valid = np.concatenate((lr_mats_valid[:,0,:,:], np.concatenate((hr_mats_valid[:,0,:,:],np.zeros((hr_mats_valid.shape[0],12,28))), axis=1)), axis=2)
    test = np.concatenate((lr_mats_test[:,0,:,:], np.concatenate((hr_mats_test[:,0,:,:],np.zeros((hr_mats_test.shape[0],12,28))), axis=1)), axis=2)

    np.savez(train_dir+f"index_train_ratio{data_ratio}.npz", inds=np.array(lr_coordinates_train, dtype=np.int_),inds_target=np.array(hr_coordinates_train, dtype=np.int_))
    np.savez(valid_dir+f"index_valid_ratio{data_ratio}.npz", inds=np.array(lr_coordinates_valid, dtype=np.int_),inds_target=np.array(hr_coordinates_valid, dtype=np.int_))
    np.savez(test_dir+f"index_test_ratio{data_ratio}.npz", inds=np.array(lr_coordinates_test, dtype=np.int_),inds_target=np.array(hr_coordinates_test, dtype=np.int_))

    np.save(train_dir+f"train_ratio{data_ratio}", train)
    np.save(valid_dir+f"valid_ratio{data_ratio}", valid)
    np.save(test_dir+f"test_ratio{data_ratio}", test)   
    
    
else:
    assert model == "hicplus", "    model name is not correct "
    hr_mats_train,lr_mats_train,hr_coordinates_train,lr_coordinates_train = hicplus_data_split([f'chr{idx}' for idx in list(range(1,18))]) # train:1~17
    hr_mats_test,lr_mats_test,hr_coordinates_test,lr_coordinates_test = hicplus_data_split([f'chr{idx}' for idx in list(range(18,23))]) # test:18~22
    print(f"\n  ...Done cropping whole matrix into submatrix for {model} training...")

    np.savez(train_dir+f"train_ratio{data_ratio}.npz", data=lr_mats_train,target=hr_mats_train,inds=np.array(lr_coordinates_train, dtype=np.int_),inds_target=np.array(hr_coordinates_train, dtype=np.int_))
    np.savez(test_dir+f"test_ratio{data_ratio}.npz", data=lr_mats_test,target=hr_mats_test,inds=np.array(lr_coordinates_test, dtype=np.int_),inds_target=np.array(hr_coordinates_test, dtype=np.int_))
    shutil.rmtree(test_dir)
    
    
print(f"\n  ...Generated data is saved in {save_dir}...\n")