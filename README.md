# HiHiC

*A benchmarking framework for deep learning models for Hi-C contact map enhancement.*   
*Docker images are available on Docker Hub: [https://hub.docker.com/repositories/jkrlab](https://hub.docker.com/repositories/jkrlab)*

## Overview

HiHiC provides a unified framework for:

- preprocessing Hi-C datasets
- generating training datasets
- training eight deep learning models
- enhancing low-resolution Hi-C contact maps
- reconstructing chromosome-scale contact matrices

## Supported models

- DFHiC
- DeepHiC
- HiCNN
- HiCARN1
- HiCARN2
- HiCPlus
- SRHiC
- iEnhance


## 1 Prepare the Data Generation Environment
------------------------------------


### Step 1. Clone the HiHiC repository.


```
git clone https://github.com/Biomedical-Data-Science-Laboratory/HiHiC.git
```




### Step 2. Run the preprocessing Docker image.


+ With GPU (CUDA 11.4)
```
docker run --rm --gpus all -it --name hihic_preprocess -v ${PWD}:${PWD} --user $(id -u):$(id -g) jkrlab/hihic_preprocess
```
+ Without GPU
```
docker run --rm -it --name hihic_preprocess -v ${PWD}:${PWD} --user $(id -u):$(id -g) jkrlab/hihic_preprocess
```
>Every Docker image should be run in the parent directory of HiHiC.




### Step 3. Create a symbolic link from the Juicer Tools JAR in the Docker workspace to the HiHiC directory.


```
ln -s /workspace/juicer_tools.jar /path/to/HiHiC/directory/
```



  


## 2. Generate Training and Prediction Data
-------------------------------------------------


### Step 1. Change your working directory to HiHiC.


```
cd /path/to/HiHiC/directory
```




### Step 2. Process mapped reads into a Hi-C contact map.


* Down-sample reads and generate a Hi-C contact map.


```
bash data_downsample.sh -i "./data/GSE63525_merged/GM12878_primary_merged.txt.gz" -p "GM12878_primary" -g "./hg19.txt" -r "180000000" -j "./juicer_tools.jar" -n "KR" -b "10000" -o "./data" -s "13"
```


| Argument | Description | Example |
|----------|-------------|---------|
| `-i` | Hi-C data contact reads file | `./data/GSE63525_merged/GM12878_primary_merged.txt.gz` |
| `-g` | Reference genome length file used to generate the Hi-C data | `./hg19.txt` |
| `-p` | Prefix | `GM12878_primary` |
| `-r` | Target number of reads after down-sampling<br>If set to -1, all reads will be sampled and made into a contact map. | `180000000` |
| `-j` | Path of Juicer tools | `./juicer_tools.jar` |
| `-n` | Normalization | `KR` |
| `-b` | Resolution (binning size) | `10000` | 
| `-o` | Output directory | `./data` |
| `-s` | Random seed | `13` |


>If the total number of reads is less than or equal to the requested down-sampling size, the down-sampling step is skipped and the original data are used directly.
>
>
> **Output** 
> * `{OutputDirectory}/{Prefix}/READ/{Prefix}__{ReadCount}.txt.gz`
> * `{OutputDirectory}/{Prefix}/HIC/{Prefix}__{ReadCount}.hic`
> * `{OutputDirectory}/{Prefix}/MAT/{Prefix}__{ReadCount}_{Resolution}_{Normalization}/`



### Step 3. Transform Hi-C contact maps into model input matrices.


* Generate input matrices for training.


```
bash data_generate_for_training.sh -i "./data/MAT/GM12878_primary__2946.5M_10Kb_KR/" -d "./data/MAT/GM12878_primary__180.0M_10Kb_KR/" -r "180000000" -b "10000" -m "DFHiC" -g "./hg19.txt" -o "./data_model" -s "300" -t "1 2 3 4 5 6 7 8 9 10 11 12 13 14" -v "15 16 17" -p "18 19 20 21 22" -w "8"
```


| Argument | Description | Example |
|----------|-------------|---------|
| `-i` | Directory containing chromosome interaction files (chr{N}.txt; COO format) | `./data/MAT/GM12878_primary__2946.5M_10Kb_KR/` |
| `-d` | Directory containing down-sampled chromosome interaction files (chr{N}.txt; COO format) | `./data/MAT/GM12878_primary__180.0M_10Kb_KR/` |
| `-r` | Number of down-sampled reads | `180000000` |
| `-g` | Reference genome length file corresponding to the input data | `./hg19.txt` |
| `-o` | Output directory | `./data_model` |
| `-b` | Resolution (binning size) | `10000` | 
| `-s` | Max value of Hi-C matrix | `300` |
| `-m` | Model name (DeepHiC, DFHiC, HiCARN, HiCNN, HiCPlus, iEnhance, or SRHiC; use `HiCARN` for HiCARN1/2) | `DFHiC` |
| `-t` | Chromosome numbers of training set | `"1 2 3 4 5 6 7 8 9 10 11 12 13 14"` |
| `-v` | Chromosome numbers of validation set | `"15 16 17"` |
| `-p` | Chromosome numbers of prediction set | `"18 19 20 21 22"` |
| `-w` | Number of worker processes (optional; defaults to all CPUs) | `8` |


> *In the case of HiCPlus, if validation chromosome is provided, it will be automatically incorporated into the training set.*
>
>
>**Output** 
> * `{OutputDirectory}/data_{Model}/TRAIN/{InputDataName}_{MaxValue}.npz`
> * `{OutputDirectory}/data_{Model}/VALID/{InputDataName}_{MaxValue}.npz`
> * `{OutputDirectory}/data_{Model}/TEST/{InputDataName}_{MaxValue}.npz`




* Generate input matrices for prediction.


```
bash data_generate_for_prediction.sh -i "./data/MAT/GM12878_primary__180.0M_10Kb_KR/" -b "10000" -m "DFHiC" -g "./hg19.txt" -o "./data_model" -s "250"
```
| Argument | Description | Example |
|----------|-------------|---------|
| `-i` | Directory containing chr{N}.txt files (Intra chromosome interaction in COO format) | `./data/MAT/GM12878_primary__180.0M_10Kb_KR/` |
| `-g` | Reference genome length file corresponding to the input data | `./hg19.txt` |
| `-o` | Output directory | `./data_model` |
| `-b` | Resolution (binning size) | `10000` | 
| `-s` | Max value of Hi-C matrix | `250` |
| `-m` | Model name (DeepHiC, DFHiC, HiCARN, HiCNN, HiCPlus, iEnhance, or SRHiC; use `HiCARN` for HiCARN1/2)| `DFHiC` |


> **Output** 
> * `{OutputDirectory}/data_{Model}/ENHANCEMENT/{InputDataName}.npz`






## 3. Set Up the Model Training Environment
-----------------------------------------------


### Step 1. Change your working directory to the HiHiC parent directory.


```
cd /path/to/HiHiC/parent/directory
```




### Step 2. Launch the Docker container.


* DeepHiC, HiCNN, HiCPlus, HiCARN1, HiCARN2, or iEnhance:
   + With GPU (CUDA 11.4)
   ```
   docker run --rm --gpus all -it --name hihic_torch -v ${PWD}:${PWD} --user $(id -u):$(id -g) jkrlab/hihic_torch
   ```
   + Without GPU
   ```
   docker run --rm -it --name hihic_torch -v ${PWD}:${PWD} --user $(id -u):$(id -g) jkrlab/hihic_torch
   ```


* DFHiC, or SRHiC:
   + With GPU (CUDA 11.4)
   ```
   docker run --rm --gpus all -it --name hihic_tensorflow -v ${PWD}:${PWD} --user $(id -u):$(id -g) jkrlab/hihic_tensorflow
   ```
   + Without GPU
   ```
   docker run --rm -it --name hihic_tensorflow -v ${PWD}:${PWD} --user $(id -u):$(id -g) jkrlab/hihic_tensorflow
   ```






## 4. Model Training
---------------------


### Step 1. Change your working directory to HiHiC.


```
cd /path/to/HiHiC/directory
```




### Step 2. Train the desired model.


```
bash model_train.sh -m "DFHiC" -e "500" -b "16" -g "0" -o "./checkpoints" -l "./log" -t "./data_model/data_DFHiC/TRAIN" -v "./data_model/data_DFHiC/VALID"
```


| Argument | Description | Example |
|----------|-------------|---------|
| `-m` | Model name (Available models: DeepHiC, DFHiC, HiCARN1, HiCARN2, HiCNN, HiCPlus, iEnhance, and SRHiC) | `DFHiC` |   
| `-e` | Epoch size | `500` |
| `-b` | Batch size | `16` | 
| `-g` | GPU ID  | `0` |
| `-o` | Output directory  | `./checkpoints` |   
| `-l` | Log directory | `./log` |
| `-t` | Training data directory | `./data_model/data_DFHiC/TRAIN` | 
| `-v` | Validation data directory | `./data_model/data_DFHiC/VALID` |   


> *All models are adapted from the original implementations released by their respective authors. They have been modified to provide a consistent benchmarking framework. Pretrained weights and datasets are not included to reduce repository size.*
>
>
> **Output** 
> * `{OutputDirectory}/checkpoints_{Model}/{EpochNumber}_{Time}_{Loss}` 
> * `{LossDirectory}/max_memory_usage.log`
> * `{LossDirectory}/train_loss_{Model}.log`




## 5. Hi-C Contact Map Enhancement
----------------------------------------------------------


### Step 1. Change your working directory to HiHiC.


```
cd /path/to/HiHiC/directory
```




### Step 2. Generate enhanced Hi-C contact maps.


```
bash model_prediction.sh -m "DFHiC" -c "./checkpoints/checkpoints_DFHiC/00005_0.02.38_0.0006605307.npz" -b "16" -g "0" -i "./data_model/data_DFHiC/ENHANCEMENT/GM12878__180.0M_10Kb_KR.npz" -o "./data_model_out"
```


| Argument | Description | Example |
|----------|-------------|---------|
| `-m` | Model name (Available models: DeepHiC, DFHiC, HiCARN1, HiCARN2, HiCNN, HiCPlus, iEnhance, and SRHiC) | `DFHiC` |   
| `-c` | File path of checkpoint | `./checkpoints/checkpoints_DFHiC/00005_0.02.38_0.0006605307.npz` |
| `-b` | Batch size | `16` |
| `-g` | GPU ID  | `0` |
| `-i` | File path of input data | `./data_model/data_DFHiC/ENHANCEMENT/GM12878__180.0M_10Kb_KR.npz` |
| `-o` | Output directory | `./data_model_out` |


> *For SRHiC, the checkpoint must be provided in `.meta` format.*
>
>
> **Output** 
> * `{OutputDirectory}/OUTPUT/{InputDataName}_{Model}_{Epoch}.npz`




### Step 3. Reconstruct chromosome-scale contact matrices from the predicted submatrices (except for iEnhance).


```
python data_make_whole.py -i ./data_model_out/OUTPUT/GM12878__180.0M_10Kb_KR_DFHiC_00005ep.npz -o ./data_model_out_whole -w 8
```


| Argument | Description | Example |
|----------|-------------|---------|
| `-i` | File path of submatrix data or prediction output directory | `./data_model_out/OUTPUT/GM12878__180.0M_10Kb_KR_DFHiC_00005ep.npz` or `./data_model_out/OUTPUT` |
| `-o` | Output directory | `./data_model_out_whole` |
| `-w` | Number of worker processes (optional; defaults to the number of available CPU cores) | `8` |


> *For iEnhance, chromosome-scale contact matrices are generated during prediction, so this step can be skipped.*
>
>
> **Output** 
> * `{OutputDirectory}/{Prefix}/{InputDataName}`
