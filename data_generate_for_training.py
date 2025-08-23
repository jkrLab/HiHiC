import os, sys, math, random, argparse, shutil
import numpy as np
from multiprocessing import Pool
import tempfile

path = os.getcwd()
random.seed(100)

# Lightweight normalization: clip to max_value and scale to [0,1] per-matrix using float32
def _normalize_matrix_inplace(matrix: np.ndarray, max_value: int) -> None:
    # Ensure float32 to reduce memory
    if matrix.dtype != np.float32:
        matrix[:] = matrix.astype(np.float32, copy=False)
    # Clip and scale in place
    np.clip(matrix, 0, max_value, out=matrix)
    local_max = float(matrix.max())
    if local_max > 0.0:
        matrix /= np.float32(local_max)

# 인자 받기
parser = argparse.ArgumentParser(description='Read Hi-C contact map and Divide submatrix for training and prediction', add_help=True)
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('-i', '--input_data_dir', dest='input_data_dir', type=str, required=True, help='Input data directory: /HiHiC/data/MAT/GM12878_163.0M_10Kb_KR/')
required.add_argument('-d', '--input_downsample_dir', dest='input_downsample_dir',type=str, required=True, help='Downsampled input data directory: /HiHiC/data/MAT/GM12878_10.2M_10Kb_KR/')
required.add_argument('-b', '--bin_size', dest='bin_size', type=str, required=True, help='Bin size(resolution): 10000')
required.add_argument('-m', '--model', dest='model', type=str, required=True, choices=['HiCPlus', 'HiCNN', 'SRHiC', 'DeepHiC', 'HiCARN', 'DFHiC', 'iEnhance'])
required.add_argument('-g', '--ref_genome', dest='ref_genome', type=str, required=True, help='Reference genome(chromosome length): /HiHiC/hg19.txt')
required.add_argument('-o', '--output_dir', dest='output_dir', type=str, required=True, help='Parent directory of output: /HiHiC/data_model/data_DFHiC/')
required.add_argument('-s', '--max_value', dest='max_value', type=str, required=True, default='300', help='Maximum value across all matrices')
optional.add_argument('-t', '--train_set', dest='train_set', type=str, required=False, help='Train set chromosome: "1 2 3 4 5 6 7 8 9 10 11 12 13 14"')
optional.add_argument('-v', '--valid_set', dest='valid_set', type=str, required=False, help='Validation set chromosome: "15 16 17"')
optional.add_argument('-p', '--test_set', dest='test_set', type=str, required=False, help='Prediction set chromosome: "18 19 20 21 22"')
optional.add_argument('-w', '--workers', dest='workers', type=int, required=False, default=os.cpu_count(), help='Number of worker processes for building contact matrices')

args = parser.parse_args()
file_list = os.listdir(args.input_data_dir)
train_set = args.train_set.split()
valid_set = args.valid_set.split()
test_set = args.test_set.split()
chrom_list = args.train_set.split() + args.valid_set.split() + args.test_set.split()
max_value = int(args.max_value) # minmax scaling
bin_size = args.bin_size

def _build_contact_mats_for_chrom(task):
    file, res, chrom_len, input_hr_dir, input_lr_dir, max_value, temp_dir = task
    chrom = file.split('.')[0]
    mat_dim = int(math.ceil(chrom_len[chrom]*1.0/res))
    hr_contact_matrix = np.zeros((mat_dim, mat_dim), dtype=np.float32)
    lr_contact_matrix = np.zeros((mat_dim, mat_dim), dtype=np.float32)
    
    # HR
    hr_hic_file = f'{input_hr_dir}/{file}'
    with open(hr_hic_file, 'r') as f:
        for line in f:
            parts = line.rstrip().split('\t')
            if len(parts) < 3:
                continue
            idx1 = int(parts[0]); idx2 = int(parts[1]); value = float(parts[2])
            if np.isnan(value):
                continue
            i = int(idx1//res); j = int(idx2//res)
            if i>=mat_dim or j>=mat_dim:
                continue
            hr_contact_matrix[i, j] = np.float32(value)
    hr_contact_matrix += hr_contact_matrix.T - np.diag(hr_contact_matrix.diagonal())
    if np.isnan(hr_contact_matrix).any():
        print(f'hr_{chrom} has nan value!', flush=True)
    _normalize_matrix_inplace(hr_contact_matrix, max_value)
    
    # LR
    lr_hic_file = f'{input_lr_dir}/{file}'
    with open(lr_hic_file, 'r') as f:
        for line in f:
            parts = line.rstrip().split('\t')
            if len(parts) < 3:
                continue
            idx1 = int(parts[0]); idx2 = int(parts[1]); value = float(parts[2])
            if np.isnan(value):
                continue
            i = int(idx1//res); j = int(idx2//res)
            if i>=mat_dim or j>=mat_dim:
                continue
            lr_contact_matrix[i, j] = np.float32(value)
    lr_contact_matrix += lr_contact_matrix.T - np.diag(lr_contact_matrix.diagonal())
    if np.isnan(lr_contact_matrix).any():
        print(f'lr_{chrom} has nan value!', flush=True)
    _normalize_matrix_inplace(lr_contact_matrix, max_value)
    
    # Save matrices to temp files to avoid serialization issues
    hr_temp_file = os.path.join(temp_dir, f'hr_{chrom}.npy')
    lr_temp_file = os.path.join(temp_dir, f'lr_{chrom}.npy')
    np.save(hr_temp_file, hr_contact_matrix)
    np.save(lr_temp_file, lr_contact_matrix)
    
    return chrom, hr_temp_file, lr_temp_file, float(hr_contact_matrix.sum()), float(lr_contact_matrix.sum())

# 크로모좀 별로 matrix 만들기 (병렬 + 임시파일 사용)
def hic_matrix_extraction(file_list, res):
    res = int(res)
    chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open(f'{args.ref_genome}').readlines()}
    workers = max(1, int(args.workers) if args.workers else os.cpu_count())
    
    # Create temporary directory for matrix files
    temp_dir = tempfile.mkdtemp(prefix='hihic_matrices_')
    
    try:
        tasks = [(file, res, chrom_len, args.input_data_dir, args.input_downsample_dir, int(args.max_value), temp_dir) for file in file_list]
        hr_contacts_dict = {}
        lr_contacts_dict = {}
        ct_hr_contacts = {}
        ct_lr_contacts = {}
        
        with Pool(processes=min(workers, len(tasks))) as pool:
            for chrom, hr_temp_file, lr_temp_file, hr_sum, lr_sum in pool.imap_unordered(_build_contact_mats_for_chrom, tasks):
                # Load matrices from temp files
                hr_contacts_dict[chrom] = np.load(hr_temp_file)
                lr_contacts_dict[chrom] = np.load(lr_temp_file)
                ct_hr_contacts[chrom] = hr_sum
                ct_lr_contacts[chrom] = lr_sum
                print(f"Processed {chrom}", flush=True)
        
        print("\n  ...Done making whole contact map by each chromosomes...", flush=True)
        return hr_contacts_dict, lr_contacts_dict, ct_hr_contacts, ct_lr_contacts
    
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

# 매트릭스 자르기
def crop_hic_matrix_by_chrom(chrom, for_model, bin_size): 
    chr = int(chrom.split('chr')[1].split(".")[0])
    chrom = chrom.split(".")[0]
    thred=2000000/int(bin_size) # thred=2M/resolution
    distance=[]
    hr_crop_mats=[]
    lr_crop_mats=[]
    hr_coordinates=[]    
    lr_coordinates=[]
    row,col = hr_contacts_dict[chrom].shape
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
                if quality_control(lr_contacts_dict[chrom][idx1:idx1+40,idx2:idx2+40]):
                    distance.append([idx1-idx2,chrom]) # DFHiC
                    lr_coordinates.append([chr, row, idx1, idx2]) # row:HiCARN                    
                    lr_contact = lr_contacts_dict[chrom][idx1:idx1+40,idx2:idx2+40]
                    if for_model in ["HiCARN", "DeepHiC", "DFHiC"]:
                        hr_contact = hr_contacts_dict[chrom][idx1:idx1+40,idx2:idx2+40]
                        hr_coordinates.append([chr, row, idx1, idx2])
                    elif for_model in ["HiCNN", "HiCPlus"]: # output mat:28*28
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
        hr_crop_mats,lr_crop_mats,hr_coordinates,_,_ = crop_hic_matrix_by_chrom(chrom,for_model="DeepHiC",bin_size=args.bin_size)
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
        hr_crop_mats,lr_crop_mats,hr_coordinates,_,_ = crop_hic_matrix_by_chrom(chrom,for_model="HiCARN",bin_size=args.bin_size)
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
        hr_crop_mats,lr_crop_mats,hr_coordinates,_,distance = crop_hic_matrix_by_chrom(chrom,for_model="DFHiC",bin_size=args.bin_size)
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
        hr_crop_mats,lr_crop_mats,hr_coordinates,lr_coordinates,_= crop_hic_matrix_by_chrom(chrom,for_model="HiCNN",bin_size=args.bin_size)
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
        hr_crop_mats,lr_crop_mats,hr_coordinates,lr_coordinates,_ = crop_hic_matrix_by_chrom(chrom,for_model="SRHiC",bin_size=args.bin_size)
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

def HiCPlus_data_split(chrom_list):
    assert len(chrom_list)>0
    hr_mats,lr_mats,hr_coords,lr_coords=[],[],[],[]
    for chrom in chrom_list: 
        hr_crop_mats,lr_crop_mats,hr_coordinates,lr_coordinates,_ = crop_hic_matrix_by_chrom(chrom,for_model="HiCPlus",bin_size=args.bin_size)
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
hr_contacts_dict,lr_contacts_dict,ct_hr_contacts,ct_lr_contacts = hic_matrix_extraction(file_list,args.bin_size)
print(f"\n  ...Done making whole matrices...", flush=True)

train_dir = f"{args.output_dir}/TRAIN/"
valid_dir = f"{args.output_dir}/VALID/"
test_dir = f"{args.output_dir}/TEST/"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
prefix = os.path.basename(args.input_downsample_dir)

# 모델이 원하는 포멧으로 저장
if args.model == "DFHiC":
    hr_mats_train,lr_mats_train,coordinates_train,distance_train = DFHiC_data_split([f'chr{idx}' for idx in train_set])
    hr_mats_valid,lr_mats_valid,coordinates_valid,distance_valid = DFHiC_data_split([f'chr{idx}' for idx in valid_set])
    hr_mats_test,lr_mats_test,coordinates_test,distance_test = DFHiC_data_split([f'chr{idx}' for idx in test_set])
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} training...", flush=True)

    np.savez(train_dir+f"{prefix}_{max_value}", data=lr_mats_train,target=hr_mats_train,inds=np.array(coordinates_train, dtype=np.int_),distance=distance_train)
    np.savez(valid_dir+f"{prefix}_{max_value}", data=lr_mats_valid,target=hr_mats_valid,inds=np.array(coordinates_valid, dtype=np.int_),distance=distance_valid)
    np.savez(test_dir+f"{prefix}_{max_value}", data=lr_mats_test,target=hr_mats_test,inds=np.array(coordinates_test, dtype=np.int_),distance=distance_test)
    
    print(f"train set: {train_set} \nvalid set: {valid_set} \ntest set: {test_set}" , flush=True)


elif args.model == "DeepHiC":      
    hr_mats_train,lr_mats_train,coordinates_train = DeepHiC_data_split([f'chr{idx}' for idx in train_set])
    hr_mats_valid,lr_mats_valid,coordinates_valid = DeepHiC_data_split([f'chr{idx}' for idx in valid_set])
    hr_mats_test,lr_mats_test,coordinates_test = DeepHiC_data_split([f'chr{idx}' for idx in test_set])
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} training...", flush=True)

    compacts = {int(k.split('chr')[1]) : np.nonzero(v)[0] for k, v in hr_contacts_dict.items()}
    size = {item.split()[0].split('chr')[1]:int(item.strip().split()[1])for item in open(f'{args.ref_genome}').readlines()}

    np.savez(train_dir+f"{prefix}_{max_value}", data=lr_mats_train,target=hr_mats_train,inds=np.array(coordinates_train, dtype=np.int_),compacts=compacts,sizes=size)
    np.savez(valid_dir+f"{prefix}_{max_value}", data=lr_mats_valid,target=hr_mats_valid,inds=np.array(coordinates_valid, dtype=np.int_),compacts=compacts,sizes=size)
    np.savez(test_dir+f"{prefix}_{max_value}", data=lr_mats_test,target=hr_mats_test,inds=np.array(coordinates_test, dtype=np.int_),compacts=compacts,sizes=size)
    
    print(f"train set: {train_set} \nvalid set: {valid_set} \ntest set: {test_set}" , flush=True)
    
    
elif args.model == "HiCARN":          
    hr_mats_train,lr_mats_train,coordinates_train = HiCARN_data_split([f'chr{idx}' for idx in train_set])
    hr_mats_valid,lr_mats_valid,coordinates_valid = HiCARN_data_split([f'chr{idx}' for idx in valid_set])
    hr_mats_test,lr_mats_test,coordinates_test = HiCARN_data_split([f'chr{idx}' for idx in test_set])
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} training...", flush=True)

    compacts = {int(k.split('chr')[1]) : np.nonzero(v)[0] for k, v in hr_contacts_dict.items()}
    size = {item.split()[0].split('chr')[1]:int(item.strip().split()[1])for item in open(f'{args.ref_genome}').readlines()}

    np.savez(train_dir+f"{prefix}_{max_value}", data=lr_mats_train,target=hr_mats_train,inds=np.array(coordinates_train, dtype=np.int_),compacts=compacts,sizes=size)
    np.savez(valid_dir+f"{prefix}_{max_value}", data=lr_mats_valid,target=hr_mats_valid,inds=np.array(coordinates_valid, dtype=np.int_),compacts=compacts,sizes=size)
    np.savez(test_dir+f"{prefix}_{max_value}", data=lr_mats_test,target=hr_mats_test,inds=np.array(coordinates_test, dtype=np.int_),compacts=compacts,sizes=size)

    print(f"train set: {train_set} \nvalid set: {valid_set} \ntest set: {test_set}" , flush=True)
    

elif args.model == "HiCNN":     
    hr_mats_train,lr_mats_train,hr_coordinates_train,lr_coordinates_train = HiCNN_data_split([f'chr{idx}' for idx in train_set])
    hr_mats_valid,lr_mats_valid,hr_coordinates_valid,lr_coordinates_valid = HiCNN_data_split([f'chr{idx}' for idx in valid_set])
    hr_mats_test,lr_mats_test,hr_coordinates_test,lr_coordinates_test = HiCNN_data_split([f'chr{idx}' for idx in test_set])
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} training...", flush=True)

    np.savez(train_dir+f"{prefix}_{max_value}", data=lr_mats_train,target=hr_mats_train,inds=np.array(lr_coordinates_train, dtype=np.int_),inds_target=np.array(hr_coordinates_train, dtype=np.int_))
    np.savez(valid_dir+f"{prefix}_{max_value}", data=lr_mats_valid,target=hr_mats_valid,inds=np.array(lr_coordinates_valid, dtype=np.int_),inds_target=np.array(hr_coordinates_valid, dtype=np.int_))
    np.savez(test_dir+f"{prefix}_{max_value}", data=lr_mats_test,target=hr_mats_test,inds=np.array(lr_coordinates_test, dtype=np.int_),inds_target=np.array(hr_coordinates_test, dtype=np.int_))

    print(f"train set: {train_set} \nvalid set: {valid_set} \ntest set: {test_set}" , flush=True)    
    
    
elif args.model == "SRHiC":  
    hr_mats_train,lr_mats_train,hr_coordinates_train,lr_coordinates_train = SRHiC_data_split([f'chr{idx}' for idx in train_set])
    hr_mats_valid,lr_mats_valid,hr_coordinates_valid,lr_coordinates_valid = SRHiC_data_split([f'chr{idx}' for idx in valid_set])
    hr_mats_test,lr_mats_test,hr_coordinates_test,lr_coordinates_test = SRHiC_data_split([f'chr{idx}' for idx in test_set])
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} training...", flush=True)

    train = np.concatenate((lr_mats_train[:,0,:,:], np.concatenate((hr_mats_train[:,0,:,:],np.zeros((hr_mats_train.shape[0],12,28))), axis=1)), axis=2)
    valid = np.concatenate((lr_mats_valid[:,0,:,:], np.concatenate((hr_mats_valid[:,0,:,:],np.zeros((hr_mats_valid.shape[0],12,28))), axis=1)), axis=2)
    test = np.concatenate((lr_mats_test[:,0,:,:], np.concatenate((hr_mats_test[:,0,:,:],np.zeros((hr_mats_test.shape[0],12,28))), axis=1)), axis=2)

    # np.savez(train_dir+f"index_{prefix}_{max_value}.npz", inds=np.array(lr_coordinates_train, dtype=np.int_),inds_target=np.array(hr_coordinates_train, dtype=np.int_))
    # np.savez(valid_dir+f"index_{prefix}_{max_value}.npz", inds=np.array(lr_coordinates_valid, dtype=np.int_),inds_target=np.array(hr_coordinates_valid, dtype=np.int_))
    # np.savez(test_dir+f"index_{prefix}_{max_value}.npz", inds=np.array(lr_coordinates_test, dtype=np.int_),inds_target=np.array(hr_coordinates_test, dtype=np.int_))
    np.savez(train_dir+f"{prefix}_{max_value}", data=train, inds=np.array(lr_coordinates_train, dtype=np.int_),inds_target=np.array(hr_coordinates_train, dtype=np.int_))
    np.savez(valid_dir+f"{prefix}_{max_value}", data=valid, inds=np.array(lr_coordinates_valid, dtype=np.int_),inds_target=np.array(hr_coordinates_valid, dtype=np.int_))
    np.savez(test_dir+f"{prefix}_{max_value}", data=test, inds=np.array(lr_coordinates_test, dtype=np.int_),inds_target=np.array(hr_coordinates_test, dtype=np.int_))   

    print(f"train set: {train_set} \nvalid set: {valid_set} \ntest set: {test_set}" , flush=True)    
    
else:
    assert args.model == "HiCPlus", "    model name is not correct "
    hr_mats_train,lr_mats_train,hr_coordinates_train,lr_coordinates_train = HiCPlus_data_split([f'chr{idx}' for idx in train_set+valid_set])
    hr_mats_test,lr_mats_test,hr_coordinates_test,lr_coordinates_test = HiCPlus_data_split([f'chr{idx}' for idx in test_set])
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} training...", flush=True)

    np.savez(train_dir+f"{prefix}_{max_value}", data=lr_mats_train,target=hr_mats_train,inds=np.array(lr_coordinates_train, dtype=np.int_),inds_target=np.array(hr_coordinates_train, dtype=np.int_))
    np.savez(test_dir+f"{prefix}_{max_value}", data=lr_mats_test,target=hr_mats_test,inds=np.array(lr_coordinates_test, dtype=np.int_),inds_target=np.array(hr_coordinates_test, dtype=np.int_))
    shutil.rmtree(valid_dir)

    print(f"train set: {train_set + valid_set} \ntest set: {test_set}" , flush=True)    
    
print(f"\n  ...Generated data is saved in {args.output_dir}...\n", flush=True)