import os, sys, math, random, argparse, shutil
import numpy as np
from sklearn.preprocessing import MinMaxScaler

path = os.getcwd()
random.seed(100)
scaler = MinMaxScaler(feature_range=(0,1))

# 인자 받기
parser = argparse.ArgumentParser(description='Read Hi-C contact map and Divide submatrix for train and predict', add_help=True)
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')

required.add_argument('-i', '--input_data_dir', dest='input_data_dir', type=str, required=True, help='Input data directory: /HiHiC/data')
required.add_argument('-m', '--model', dest='model', type=str, required=True, choices=['HiCPlus', 'HiCNN', 'SRHiC', 'DeepHiC', 'HiCARN', 'DFHiC', 'iEnhance'])
required.add_argument('-g', '--ref_chrom', dest='ref_chrom', type=str, required=True, help='Reference chromosome length: /HiHiC/hg19.txt')
required.add_argument('-o', '--output_dir', dest='output_dir', type=str, required=True, help='Parent directory of output: /data/HiHiC/')
required.add_argument('-s', '--max_value', dest='max_value', type=str, required=True, default='300', help='Maximum value of chromosome matrix')
required.add_argument('-n', '--normalization', dest='normalization', type=str, required=True, default='None', help='Normalization method')

args = parser.parse_args()
file_list = os.listdir(args.input_data_dir)
chrom_list = [file.split('_')[0] for file in file_list]
normalization = args.normalization
max_value = int(args.max_value) # minmax scaling

# 크로모좀 별로 matrix 만들기
def hic_matrix_extraction(file_list, res=10000):
    chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open(f'{args.ref_chrom}').readlines()} # GM12878 Hg19

    contacts_dict={}
    for file in file_list:
        chr = file.split('_')[0] # 'chr1'
        lr_hic_file = f'{args.input_data_dir}/{file}'
        mat_dim = int(math.ceil(chrom_len[chr]*1.0/res))
        lr_contact_matrix = np.zeros((mat_dim,mat_dim))
        for line in open(lr_hic_file).readlines():
            idx1, idx2, value = int(line.strip().split('\t')[0]),int(line.strip().split('\t')[1]),float(line.strip().split('\t')[2])
            if np.isnan(value):
                continue
            elif idx2/res>=mat_dim or idx1/res>=mat_dim:
                continue
            else:
                lr_contact_matrix[int(idx1/res)][int(idx2/res)] = value
        lr_contact_matrix+= lr_contact_matrix.T - np.diag(lr_contact_matrix.diagonal())
        if np.isnan(lr_contact_matrix).any(): ###############
            print(f'lr_{chr} has nan value!', flush=True) ###############
        contacts_dict[chr] = scaler.fit_transform(np.minimum(max_value, lr_contact_matrix)) # (0,300) >> (0,1)

    ct_contacts={item:sum(sum(contacts_dict[item])) for item in contacts_dict.keys()} # read 수
    print("\n  ...Done making whole contact map by each chromosomes...", flush=True)
    return contacts_dict,ct_contacts

# 매트릭스 자르기
def crop_hic_matrix_by_chrom(chrom, for_model, thred=200): # thred=2M/resolution
    chr = int(chrom.split('chr')[1])
    distance=[]
    crop_mats=[]
    coordinates=[]    
    row,col = contacts_dict[chrom].shape
    if row<=thred or col<=thred: # bin 수가 200 보다 작으면 False
        print('HiC matrix size wrong!', flush=True)
        sys.exit()
    def quality_control(mat,thred=0.05):
        if len(mat.nonzero()[0])<thred*mat.shape[0]*mat.shape[1]: # 숫자 있는 셀의 수가 전체의 5% 미만이면 False
            return False
        else:
            return True
    for idx1 in range(0,row-40,28):
        for idx2 in range(0,col-40,28):
            if abs(idx1-idx2)<thred:
                if quality_control(contacts_dict[chrom][idx1:idx1+40,idx2:idx2+40]):
                    distance.append([idx1-idx2,chrom]) # DFHiC
                    if for_model in ["HiCARN", "DeepHiC", "DFHiC"]:
                        contact = contacts_dict[chrom][idx1:idx1+40,idx2:idx2+40]
                        coordinates.append([chr, row, idx1, idx2])
                    elif for_model in ["HiCNN", "HiCPlus"]: # output mat:28*28
                        contact = contacts_dict[chrom][idx1:idx1+40,idx2:idx2+40]
                        coordinates.append([chr, row, idx1+6, idx2+6])
                    else:
                        assert for_model in ["SRHiC"]
                        contact = contacts_dict[chrom][idx1+6:idx1+34,idx2+6:idx2+34]
                        coordinates.append([chr, row, idx1+6, idx2+6]) 
                    crop_mats.append(contact)
    crop_mats                
    crop_mats = np.concatenate([item[np.newaxis,:] for item in crop_mats],axis=0)
    return crop_mats,coordinates,distance     


# 모델별 submatrix 생성
def DeepHiC_data_split(chrom_list):
    assert len(chrom_list)>0
    lr_mats,lr_coords=[],[]
    for chrom in chrom_list:
        lr_crop_mats,lr_coordinates,_ = crop_hic_matrix_by_chrom(chrom,for_model="DeepHiC",thred=200)
        lr_mats.append(lr_crop_mats)
        lr_coords.append(lr_coordinates)
    lr_mats = np.concatenate(lr_mats,axis=0)
    lr_mats=lr_mats[:,np.newaxis]
    lr_coords = sum(lr_coords, [])
    return lr_mats,lr_coords

def HiCARN_data_split(chrom_list):
    assert len(chrom_list)>0
    lr_mats,lr_coords=[],[]
    for chrom in chrom_list:
        lr_crop_mats,lr_coordinates,_ = crop_hic_matrix_by_chrom(chrom,for_model="HiCARN",thred=200)
        lr_mats.append(lr_crop_mats)
        lr_coords.append(lr_coordinates)
    lr_mats = np.concatenate(lr_mats,axis=0)
    lr_mats=lr_mats[:,np.newaxis]
    lr_coords = sum(lr_coords, [])
    return lr_mats,lr_coords

def DFHiC_data_split(chrom_list):
    distance_all=[]
    assert len(chrom_list)>0
    lr_mats,lr_coords=[],[]
    for chrom in chrom_list:
        lr_crop_mats,lr_coordinates,distance = crop_hic_matrix_by_chrom(chrom,for_model="DFHiC",thred=200)
        distance_all+=distance
        lr_mats.append(lr_crop_mats)
        lr_coords.append(lr_coordinates)
    lr_mats = np.concatenate(lr_mats,axis=0)
    lr_mats=lr_mats[:,np.newaxis]
    lr_mats=lr_mats.transpose((0,2,3,1))
    lr_coords = sum(lr_coords, [])
    return lr_mats,lr_coords,distance_all

def HiCNN_data_split(chrom_list):
    assert len(chrom_list)>0
    lr_mats,lr_coords=[],[]
    for chrom in chrom_list:
        lr_crop_mats,lr_coordinates,_= crop_hic_matrix_by_chrom(chrom,for_model="HiCNN",thred=200)
        lr_mats.append(lr_crop_mats)
        lr_coords.append(lr_coordinates)
    lr_mats = np.concatenate(lr_mats,axis=0)
    lr_mats=lr_mats[:,np.newaxis]
    lr_coords = sum(lr_coords, [])
    return lr_mats,lr_coords

def SRHiC_data_split(chrom_list):
    assert len(chrom_list)>0
    lr_mats,lr_coords=[],[]
    for chrom in chrom_list:
        lr_crop_mats,lr_coordinates,_ = crop_hic_matrix_by_chrom(chrom,for_model="SRHiC",thred=200)
        lr_mats.append(lr_crop_mats)
        lr_coords.append(lr_coordinates)
    lr_mats = np.concatenate(lr_mats,axis=0)
    lr_mats=lr_mats[:,np.newaxis]
    lr_coords = sum(lr_coords, [])
    return lr_mats,lr_coords

def HiCPlus_data_split(chrom_list):
    assert len(chrom_list)>0
    lr_mats,lr_coords=[],[]
    for chrom in chrom_list: 
        lr_crop_mats,lr_coordinates,_ = crop_hic_matrix_by_chrom(chrom,for_model="HiCPlus",thred=200)
        lr_mats.append(lr_crop_mats)
        lr_coords.append(lr_coordinates)
    lr_mats = np.concatenate(lr_mats,axis=0)
    lr_mats=lr_mats[:,np.newaxis]
    lr_coords = sum(lr_coords, [])
    return lr_mats,lr_coords


# 함수 실행
contacts_dict, ct_contacts = hic_matrix_extraction(file_list)
print(f"\n  ...Done making whole matrices...", flush=True)


save_dir = f'{args.output_dir}data_{args.model}/'
os.makedirs(save_dir, exist_ok=True)


# 모델이 원하는 포멧으로 저장
if args.model == "DFHiC":
    mats,coordinates,distance = DFHiC_data_split(chrom_list)
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} training...", flush=True)
    np.savez(save_dir+f"for_enhancement_{args.model}.npz", data=mats, inds=np.array(coordinates, dtype=np.int_),distance=distance)


elif args.model == "DeepHiC":      
    mats,coordinates = DeepHiC_data_split(chrom_list)
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} training...", flush=True)
    compacts = {int(k.split('chr')[1]) : np.nonzero(v)[0] for k, v in contacts_dict.items()}
    size = {item.split()[0].split('chr')[1]:int(item.strip().split()[1])for item in open(f'{args.ref_chrom}').readlines()}
    np.savez(save_dir+f"for_enhancement_{args.model}.npz", data=mats,inds=np.array(coordinates, dtype=np.int_),compacts=compacts,sizes=size)
    
    
elif args.model == "HiCARN":          
    mats,coordinates = HiCARN_data_split(chrom_list)
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} training...", flush=True)
    compacts = {int(k.split('chr')[1]) : np.nonzero(v)[0] for k, v in contacts_dict.items()}
    size = {item.split()[0].split('chr')[1]:int(item.strip().split()[1])for item in open(f'{args.ref_chrom}').readlines()}
    np.savez(save_dir+f"for_enhancement_{args.model}.npz", data=mats,inds=np.array(coordinates, dtype=np.int_),compacts=compacts,sizes=size)
   

elif args.model == "HiCNN":     
    mats,coordinates = HiCNN_data_split(chrom_list)
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} training...", flush=True)
    np.savez(save_dir+f"for_enhancement_{args.model}.npz", data=mats,inds=np.array(coordinates, dtype=np.int_))


elif args.model == "SRHiC":  
    mats,coordinates = SRHiC_data_split(chrom_list)
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} training...", flush=True)
    mats = np.concatenate((mats[:,0,:,:], np.concatenate((mats[:,0,:,:],np.zeros((mats.shape[0],12,28))), axis=1)), axis=2)
    np.savez(save_dir+f"index_for_enhancement_{args.model}.npz", inds=np.array(coordinates, dtype=np.int_))
    np.save(save_dir+f"for_enhancement_{args.model}", mats)
    

else:
    assert args.model == "HiCPlus", "    model name is not correct "
    mats,coordinates = HiCPlus_data_split(chrom_list)
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} training...", flush=True)
    np.savez(save_dir+f"for_enhancement_{args.model}.npz", data=mats,inds=np.array(coordinates, dtype=np.int_))

    
print(f"\n  ...Generated data is saved in {save_dir}...\n", flush=True)