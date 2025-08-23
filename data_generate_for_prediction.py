import os, sys, math, random, argparse, shutil
import numpy as np
from multiprocessing import Pool
import tempfile

path = os.getcwd()
random.seed(100)

# Lightweight normalization: clip to max_value and scale to [0,1] per-matrix using float32
def _normalize_matrix_inplace(matrix: np.ndarray, max_value: int) -> None:
    if matrix.dtype != np.float32:
        matrix[:] = matrix.astype(np.float32, copy=False)
    np.clip(matrix, 0, max_value, out=matrix)
    local_max = float(matrix.max())
    if local_max > 0.0:
        matrix /= np.float32(local_max)

# 인자 받기
parser = argparse.ArgumentParser(description='Read Hi-C contact map and Divide submatrix for training and prediction', add_help=True)
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('-i', '--input_data_dir', dest='input_data_dir', type=str, required=True, help='Input data directory: /HiHiC/data/MAT/GM12878_10.2M_10Kb_KR')
required.add_argument('-b', '--bin_size', dest='bin_size', type=str, required=True, help='Bin size(resolution): 10000')
required.add_argument('-m', '--model', dest='model', type=str, required=True, choices=['HiCPlus', 'HiCNN', 'SRHiC', 'DeepHiC', 'HiCARN', 'DFHiC', 'iEnhance'])
required.add_argument('-g', '--ref_genome', dest='ref_genome', type=str, required=True, help='Reference chromosome length: /HiHiC/hg19.txt')
required.add_argument('-o', '--output_dir', dest='output_dir', type=str, required=True, help='Parent directory of output: /HiHiC/data_model/data_DFHiC/')
required.add_argument('-s', '--max_value', dest='max_value', type=str, required=True, default='300', help='Maximum value across all matrices')
optional.add_argument('-w', '--workers', dest='workers', type=int, required=False, default=os.cpu_count(), help='Number of worker processes for building contact matrices')

args = parser.parse_args()
file_list = os.listdir(args.input_data_dir)
chrom_list = [file.split('.')[0] for file in file_list]
max_value = int(args.max_value) # minmax scaling
bin_size = args.bin_size

def _build_contact_mat_for_chrom(task):
    file, res, chrom_len, input_dir, max_value, temp_dir = task
    chrom = file.split('.')[0]
    mat_dim = int(math.ceil(chrom_len[chrom]*1.0/res))
    lr_contact_matrix = np.zeros((mat_dim, mat_dim), dtype=np.float32)
    lr_hic_file = f'{input_dir}/{file}'
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
    
    # Save matrix to temp file to avoid serialization issues
    lr_temp_file = os.path.join(temp_dir, f'lr_{chrom}.npy')
    np.save(lr_temp_file, lr_contact_matrix)
    
    return chrom, lr_temp_file, float(lr_contact_matrix.sum())

# 크로모좀 별로 matrix 만들기 (병렬 + 임시파일 사용)
def hic_matrix_extraction(file_list, bin_size):
    res=int(bin_size)
    chrom_len = {item.split()[0]:int(item.strip().split()[1]) for item in open(f'{args.ref_genome}').readlines()} # GM12878 Hg19

    workers = max(1, int(args.workers) if args.workers else os.cpu_count())
    
    # Create temporary directory for matrix files
    temp_dir = tempfile.mkdtemp(prefix='hihic_prediction_')
    
    try:
        tasks = [(file, res, chrom_len, args.input_data_dir, int(args.max_value), temp_dir) for file in file_list]
        contacts_dict = {}
        ct_contacts = {}
        with Pool(processes=min(workers, len(tasks))) as pool:
            for chrom, lr_temp_file, lr_sum in pool.imap_unordered(_build_contact_mat_for_chrom, tasks):
                # Load matrix from temp file
                contacts_dict[chrom] = np.load(lr_temp_file)
                ct_contacts[chrom] = lr_sum
                print(chrom, ":", contacts_dict[chrom].shape)
        print("\n  ...Done making whole contact map by each chromosomes...", flush=True)
        return contacts_dict,ct_contacts
    
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)

def crop_hic_matrix_by_chrom(chrom, for_model, bin_size): 
    # Dynamic chromosome mapping based on available chromosomes
    chrom_num = chrom.split('chr')[1].split(".")[0]
    
    # Get all available chromosomes and create mapping
    available_chroms = sorted(contacts_dict.keys())
    chrom_to_num = {}
    for i, ch in enumerate(available_chroms):
        chrom_to_num[ch] = i + 1
    
    chr = chrom_to_num[chrom]
    chrom = chrom.split(".")[0]
    thred=2000000/int(bin_size) # thred=2M/resolution 
    distance=[] # DFHiC
    lr_crop_mats=[]
    hr_coordinates=[]    
    lr_coordinates=[] # HiCARN
    row,col = contacts_dict[chrom].shape
    if row<=thred or col<=thred: # bin 수가 200 보다 작으면 False
        print('HiC matrix size wrong!', flush=True)
        sys.exit()
    # def quality_control(mat,thred=0.05):
    #     if len(mat.nonzero()[0])<thred*mat.shape[0]*mat.shape[1]: # 숫자 있는 셀의 수가 전체의 5% 미만이면 False (분석 가능한 수준 설정)
    #         return False
    #     else:
    #         return True 
    for idx1 in range(0,row-40,28):
        for idx2 in range(0,col-40,28):
             if abs(idx1-idx2)<thred:
                distance.append([idx1-idx2,chrom]) # DFHiC
                lr_coordinates.append([chr, row, idx1, idx2]) # row:HiCARN                    
                lr_contact = contacts_dict[chrom][idx1:idx1+40,idx2:idx2+40]
                if for_model in ["HiCARN", "DeepHiC", "DFHiC"]:
                    hr_coordinates.append([chr, row, idx1, idx2])
                elif for_model in ["HiCNN", "HiCPlus"]: # output mat:28*28
                    hr_coordinates.append([chr, row, idx1+6, idx2+6])
                else:
                    assert for_model in ["SRHiC"]
                    hr_coordinates.append([chr, row, idx1+6, idx2+6]) 
                lr_crop_mats.append(lr_contact)
    lr_crop_mats = np.concatenate([item[np.newaxis,:] for item in lr_crop_mats],axis=0)
    return lr_crop_mats,hr_coordinates,lr_coordinates,distance     


# 모델별 submatrix 생성
def DeepHiC_data_split(chrom_list):
    assert len(chrom_list)>0
    lr_mats,hr_coords=[],[]
    for chrom in chrom_list:
        lr_crop_mats,hr_coordinates,_,_ = crop_hic_matrix_by_chrom(chrom,for_model="DeepHiC",bin_size=args.bin_size)
        lr_mats.append(lr_crop_mats)
        hr_coords.append(hr_coordinates)
    lr_mats = np.concatenate(lr_mats,axis=0)
    lr_mats=lr_mats[:,np.newaxis]
    hr_coords = sum(hr_coords, [])
    return lr_mats,hr_coords

def HiCARN_data_split(chrom_list):
    assert len(chrom_list)>0
    lr_mats,hr_coords=[],[]
    for chrom in chrom_list:
        lr_crop_mats,hr_coordinates,_,_ = crop_hic_matrix_by_chrom(chrom,for_model="HiCARN",bin_size=args.bin_size)
        lr_mats.append(lr_crop_mats)
        hr_coords.append(hr_coordinates)
    lr_mats = np.concatenate(lr_mats,axis=0)
    lr_mats=lr_mats[:,np.newaxis]
    hr_coords = sum(hr_coords, [])
    return lr_mats,hr_coords

def DFHiC_data_split(chrom_list):
    distance_all=[]
    assert len(chrom_list)>0
    lr_mats,hr_coords=[],[]
    for chrom in chrom_list:
        lr_crop_mats,hr_coordinates,_,distance = crop_hic_matrix_by_chrom(chrom,for_model="DFHiC",bin_size=args.bin_size)
        distance_all+=distance
        lr_mats.append(lr_crop_mats)
        hr_coords.append(hr_coordinates)
    lr_mats = np.concatenate(lr_mats,axis=0)
    lr_mats=lr_mats[:,np.newaxis]
    lr_mats=lr_mats.transpose((0,2,3,1))
    hr_coords = sum(hr_coords, [])
    return lr_mats,hr_coords,distance_all

def HiCNN_data_split(chrom_list):
    assert len(chrom_list)>0
    lr_mats,hr_coords,lr_coords=[],[],[]
    for chrom in chrom_list:
        lr_crop_mats,hr_coordinates,lr_coordinates,_= crop_hic_matrix_by_chrom(chrom,for_model="HiCNN",bin_size=args.bin_size)
        lr_mats.append(lr_crop_mats)
        hr_coords.append(hr_coordinates)
        lr_coords.append(lr_coordinates)
    lr_mats = np.concatenate(lr_mats,axis=0)
    lr_mats=lr_mats[:,np.newaxis]
    hr_coords = sum(hr_coords, [])
    lr_coords = sum(lr_coords, [])
    return lr_mats,hr_coords,lr_coords

def SRHiC_data_split(chrom_list):
    assert len(chrom_list)>0
    lr_mats,hr_coords,lr_coords=[],[],[]
    for chrom in chrom_list:
        lr_crop_mats,hr_coordinates,lr_coordinates,_ = crop_hic_matrix_by_chrom(chrom,for_model="SRHiC",bin_size=args.bin_size)
        lr_mats.append(lr_crop_mats)
        hr_coords.append(hr_coordinates)
        lr_coords.append(lr_coordinates)
    lr_mats = np.concatenate(lr_mats,axis=0)
    lr_mats=lr_mats[:,np.newaxis]
    hr_coords = sum(hr_coords, [])
    lr_coords = sum(lr_coords, [])
    return lr_mats,hr_coords,lr_coords

def HiCPlus_data_split(chrom_list):
    assert len(chrom_list)>0
    lr_mats,hr_coords,lr_coords=[],[],[]
    for chrom in chrom_list: 
        lr_crop_mats,hr_coordinates,lr_coordinates,_ = crop_hic_matrix_by_chrom(chrom,for_model="HiCPlus",bin_size=args.bin_size)
        lr_mats.append(lr_crop_mats)
        hr_coords.append(hr_coordinates)
        lr_coords.append(lr_coordinates)
    lr_mats = np.concatenate(lr_mats,axis=0)
    lr_mats=lr_mats[:,np.newaxis]
    hr_coords = sum(hr_coords, [])
    lr_coords = sum(lr_coords, [])
    return lr_mats,hr_coords,lr_coords


# 함수 실행
contacts_dict,ct_contacts = hic_matrix_extraction(file_list, bin_size)
print(f"\n  ...Done making whole matrices...", flush=True)

saved_in = os.path.join(args.output_dir,"ENHANCEMENT")
os.makedirs(saved_in, exist_ok=True)
prefix = os.path.basename(args.input_data_dir)
out_file = os.path.join(saved_in, f"{prefix}")

# 모델이 원하는 포멧으로 저장
if args.model == "DFHiC":
    mats,coordinates,distance = DFHiC_data_split(chrom_list)
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} prediction...", flush=True)
    np.savez(out_file, data=mats, inds=np.array(coordinates, dtype=np.int_),distance=distance)


elif args.model == "DeepHiC":      
    mats,coordinates = DeepHiC_data_split(chrom_list)
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} prediction...", flush=True)
    # Dynamic chromosome mapping for compacts
    available_chroms = sorted(contacts_dict.keys())
    chrom_to_num = {}
    for i, ch in enumerate(available_chroms):
        chrom_to_num[ch] = i + 1
    
    compacts = {chrom_to_num[k] : np.nonzero(v)[0] for k, v in contacts_dict.items()}
    size = {}
    for item in open(f'{args.ref_genome}').readlines():
        chrom_name = item.split()[0]
        size_val = int(item.strip().split()[1])
        # Keep original chromosome name (X, Y) for size dictionary
        chrom_key = chrom_name.split('chr')[1]
        size[chrom_key] = size_val
    np.savez(out_file, data=mats,inds=np.array(coordinates, dtype=np.int_),compacts=compacts,sizes=size)
    
    
elif args.model == "HiCARN":          
    mats,coordinates = HiCARN_data_split(chrom_list)
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} prediction...", flush=True)
    # Dynamic chromosome mapping for compacts
    available_chroms = sorted(contacts_dict.keys())
    chrom_to_num = {}
    for i, ch in enumerate(available_chroms):
        chrom_to_num[ch] = i + 1
    
    compacts = {chrom_to_num[k] : np.nonzero(v)[0] for k, v in contacts_dict.items()}
    size = {}
    for item in open(f'{args.ref_genome}').readlines():
        chrom_name = item.split()[0]
        size_val = int(item.strip().split()[1])
        # Keep original chromosome name (X, Y) for size dictionary
        chrom_key = chrom_name.split('chr')[1]
        size[chrom_key] = size_val
    np.savez(out_file, data=mats,inds=np.array(coordinates, dtype=np.int_),compacts=compacts,sizes=size)
   

elif args.model == "HiCNN":     
    mats,hr_coords,coords = HiCNN_data_split(chrom_list)
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} prediction...", flush=True)
    np.savez(out_file, data=mats,inds=np.array(coords, dtype=np.int_),inds_target=np.array(hr_coords, dtype=np.int_))


elif args.model == "SRHiC":  
    mats,hr_coords,coords = SRHiC_data_split(chrom_list)
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} prediction...", flush=True)
    mats = mats[:,0,:,:]
    # np.savez(os.path.join(saved_in, f"index_{prefix}.npz"), inds=np.array(coords, dtype=np.int_), inds_target=np.array(hr_coords, dtype=np.int_))
    np.savez(out_file, data=mats, inds=np.array(coords, dtype=np.int_),inds_target=np.array(hr_coords, dtype=np.int_))
    

else:
    assert args.model == "HiCPlus", "    model name is not correct "
    mats,hr_coords,coords = HiCPlus_data_split(chrom_list)
    print(f"\n  ...Done cropping whole matrix into submatrix for {args.model} prediction...", flush=True)
    np.savez(out_file, data=mats,inds=np.array(coords, dtype=np.int_),inds_target=np.array(hr_coords, dtype=np.int_))

    
print(f"\n  ...Generated data is saved in {args.output_dir}...\n", flush=True)