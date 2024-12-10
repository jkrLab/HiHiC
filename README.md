HiHiC
=====
*A highlighting platform comparing deep learning models for Hi-C contact map enhancement*   
*HiHiC docker repository: [https://hub.docker.com/repositories/jkrlab](https://hub.docker.com/repositories/jkrlab)*




Ⅰ. Environment for data generation
------------------------------------


1. Clone HiHiC repository


```
git clone https://github.com/jkrLab/HiHiC.git
```




2. Run the docker image for data processing


+ With GPU (CUDA 11.4)
```
docker run --rm --gpus all -it --name hihic_preprocess -v ${PWD}:${PWD} jkrlab/hihic_preprocess
```
+ Without GPU
```
docker run --rm -it --name hihic_preprocess -v ${PWD}:${PWD} jkrlab/hihic_preprocess
```
>Every docker image should be run in the parent directory of HiHiC.




3. Make a symbolic link to Juicer tools in the docker workspace to the HiHiC directory


```
ln -s /workspace/juicer_tools.jar /path/to/HiHiC/directory/
```



  


Ⅱ. Data generation for each deep learning model
-------------------------------------------------


1. Change your working directory to HiHiC


```
cd /path/to/HiHiC/directory
```




2. Process mapped read data and transfor into Hi-C contact map


* Download data from web and transform into Hi-C contact map


```
bash data_download_downsample.sh -i "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551550/suppl/GSM1551550_HIC001_merged_nodups.txt.gz" -p "GM12878" -g "./hg19.txt" -r "10187283" -j "./juicer_tools.jar" -n "KR" -b "10000" -o "./data"
```


| Argument | Description | Example |
|----------|-------------|---------|
| `-i` | Download URL of Hi-C data contact reads | `https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551550/suppl/GSM1551550_HIC001_merged_nodups.txt.gz` |
| `-g` | Reference genome length file, your Hi-C data is based on | `./hg19.txt` |
| `-p` | Prefix | `GM12878` |
| `-r` | Downsample reads<br>If set to -1, all reads will be sampled and made into a contact map. | `10187283` |
| `-j` | Path of Juicer tools | `./juicer_tools.jar` |
| `-n` | Normalization | `KR` |
| `-b` | Resolution (binning sizes) | `10000` | 
| `-o` | Directory path of output data | `./data` |


>If the total number of reads in the downloaded file is less than or equal to the downsampling read number, 
>the downsampling step and its subsequent processes will be skipped.
> Output: 
> * {OutputDirectory}/HIC/{Prefix}__{ReadsNumber}.hic
> * {OutputDirectory}/READ/{Prefix}__{ReadsNumber}.txt.gz


* Own data to downsample and transform into Hi-C contact map


```
bash data_downsample.sh -i "./data/GM12878/READ/GSM1551550_HIC001.txt.gz" -p "GM12878" -g "./hg19.txt" -r "10187283" -j "./juicer_tools.jar" -n "KR" -b "10000" -o "./data"
```


| Argument | Description | Example |
|----------|-------------|---------|
| `-i` | Hi-C data contact reads file | `./data/GM12878/READ/GSM1551550_HIC001.txt.gz` |
| `-g` | Reference genome length file, your Hi-C data is based on | `./hg19.txt` |
| `-p` | Prefix | `GM12878` |
| `-r` | Number of Downsample reads<br>If set to -1, all reads will be sampled and made into a contact map. | `10187283` |
| `-j` | Path of Juicer tools | `./juicer_tools.jar` |
| `-n` | Normalization | `KR` |
| `-b` | Resolution (binning size) | `10000` | 
| `-o` | Directory path of output data | `./data` |


>If the total number of reads in the downloaded file is less than or equal to the downsampling read number, 
>the downsampling step and its subsequent processes will be skipped.
> Output: 
> * {OutputDirectory}/MAT/{Prefix}__{ReadsNumber}_{Resolution}_{Normalization}/



3. Transform Hi-C contact map into input matrix of each model


* Input matrix for training model


```
bash data_generate_for_training.sh -i "./data/MAT/GM12878__16.3M_10Kb_KR/" -d "./data/MAT/GM12878__10.2M_10Kb_KR/" -b "10000" -m "DFHiC" -g "./hg19.txt" -o "./data_model" -s "300" -t "1 2 3 4 5 6 7 8 9 10 11 12 13 14" -v "15 16 17" -p "18 19 20 21 22"
```


| Argument | Description | Example |
|----------|-------------|---------|
| `-i` | Directory containing chr{N}.txt files (Intra chromosome interaction in COO format) | `./data/MAT/GM12878__16.3M_10Kb_KR/` |
| `-d` | Directory containing chr{N}.txt files (Intra chromosome interaction in COO format) | `./data/MAT/GM12878__10.2M_10Kb_KR/` |
| `-g` | Reference genome length file, your input data are based on | `./hg19.txt` |
| `-o` | Directory path of output data | `./data_model` |
| `-b` | Resolution (Binning size) | `10000` | 
| `-s` | Max value of Hi-C matrix | `300` |
| `-m` | Model name that you use (One of HiCARN, DeepHiC, HiCNN, DFHiC, HiCPlus, SRHiC, or iEnhance) | `DFHiC` |
| `-t` | Chromosome numbers of training set | `"1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17"` |
| `-v` | Chromosome numbers of validation set | `"15 16 17"` |
| `-p` | Chromosome numbers of prediction set | `"18 19 20 21 22"` |


> *In the case of HiCPlus, if validation chromosome is provided, it will be automatically incorporated into the training set.*
>Output: 
> * {OutputDirectory}/data_{Model}/TRAIN/{InputName}_{MaxValue}.npz
> * {OutputDirectory}/data_{Model}/VALID/{InputName}_{MaxValue}.npz
> * {OutputDirectory}/data_{Model}/TEST/{InputName}_{MaxValue}.npz




* Input matrix for model prediction (Enhancement with pretrained model)


```
bash data_generate_for_prediction.sh -i "./data/MAT/GM12878__10.2M_10Kb_KR/" -b "10000" -m "DFHiC" -g "./hg19.txt" -o "./data_model" -s "300"
```
| Argument | Description | Example |
|----------|-------------|---------|
| `-i` | Directory containing chr{N}.txt files (Intra chromosome interaction in COO format) | `./data/MAT/GM12878__10.2M_10Kb_KR/` |
| `-g` | Reference genome length file, your input data is based on | `./hg19.txt` |
| `-o` | Directory path of output data | `./data_model` |
| `-b` | Resolution (Binning size) | `10000` | 
| `-s` | Max value of Hi-C matrix | `300` |
| `-m` | Model name that you use (One of HiCARN, DeepHiC, HiCNN, DFHiC, HiCPlus, SRHiC, or iEnhance) | `DFHiC` |


> Output: 
> * {OutputDirectory}/data_{Model}/ENHANCEMENT/{InputName}_{MaxValue}.npz






Ⅲ. Environment for each deep learning model
-----------------------------------------------


1. Change your working directory to HiHiC parent directory


```
cd /path/to/HiHiC/parent/directory
```




2. Run Docker environment


* HiCPlus, HiCNN, DeepHiC, HiCARN, or iEnhance:
   + With GPU (CUDA 11.4)
   ```
   docker run --rm --gpus all -it --name hihic_torch -v ${PWD}:${PWD} jkrlab/hihic_torch
   ```
   + Without GPU
   ```
   docker run --rm -it --name hihic_torch -v ${PWD}:${PWD} jkrlab/hihic_torch
   ```


* SRHiC or DFHiC:
   + With GPU (CUDA 11.4)
   ```
   docker run --rm --gpus all -it --name hihic_tensorflow -v ${PWD}:${PWD} jkrlab/hihic_tensorflow
   ```
   + Without GPU
   ```
   docker run --rm -it --name hihic_tensorflow -v ${PWD}:${PWD} jkrlab/hihic_tensorflow
   ```






Ⅳ. Model training
---------------------


1. Change your working directory to HiHiC


```
cd /path/to/HiHiC/directory
```




2. Train the model you want with options 


```
bash model_train.sh -m "DFHiC" -e "500" -b "16" -g "0" -o "./checkpoints" -l "./log" -t "./data_model/data_DFHiC/TRAIN" -v "./data_model/data_DFHiC/VALID"
```


| Argument | Description | Example |
|----------|-------------|---------|
| `-m` | Name of the model (One of HiCARN1, HiCARN12, DeepHiC, HiCNN2, DFHiC, HiCPlus, SRHiC, or iEnhance) | `DFHiC` |   
| `-e` | Number of train epoch | `500` |
| `-b` | Number of batch size | `16` | 
| `-g` | Number of GPU ID  | `0` |
| `-o` | Directory path of output models  | `./checkpoints` |   
| `-l` | Directory path of log | `./log` |
| `-t` | Directory path of training data | `./data_model/data_DFHiC/TRAIN` | 
| `-v` | Directory path of validation data | `./data_model/data_DFHiC/VALID` |   


> *All the deep learning model codes were downloaded from each author's GitHub and modified for performance comparison. For light memory storage, pre-trained weights and data have been removed.*
> Output: 
> * {OutputDirectory}/checkpoints_{Model}/{Epoch}_{Time}_{Loss}
> * {LossDirectory}/max_memory_usage.log




Ⅴ. HiC contact map enhancement with pretrained weights
----------------------------------------------------------


1. Change your working directory to HiHiC


```
cd /path/to/HiHiC/directory
```




2. Enhance the low resolution data you have


```
bash model_prediction.sh -m "DFHiC" -c "./checkpoints/checkpoints_DFHiC/00005_0.02.38_0.0006605307.npz" -b "16" -g "0" -i "./data_model/data_DFHiC/test/test_ratio16.npz" -o "./data_model_out"
```


| Argument | Description | Example |
|----------|-------------|---------|
| `-m` | Name of the model (One of HiCARN1, HiCARN12, DeepHiC, HiCNN2, DFHiC, HiCPlus, SRHiC, or iEnhance) | `DFHiC` |   
| `-c` | File path of checkpoint | `./checkpoints/checkpoints_DFHiC/00005_0.02.38_0.0006605307.npz` |
| `-b` | Number of batch size | `16` |
| `-g` | Number of GPU ID  | `0` |
| `-i` | File path of input data | `./data_DFHiC/ENHANCEMENT/GM12878__2.0M_10Kb_KR.npz` |
| `-o` | Directory path of output data | `./data_model_out` |


> *When you use SRHiC, the checkpoint file need .meta format.*
> Output: 
> * {OutputDirectory}/OUTPUT/{InputName}_{Model}_{Epoch}.npz




3. Create chromosom matrix with enhanced submatrix (Except for iEnhance)


```
python data_make_whole.py -i ./data_model_out/OUTPUT/GM12878__2.0M_10Kb_KR_DFHiC_00005ep.npz  -o ./data_model_out
```


| Argument | Description | Example |
|----------|-------------|---------|
| `-i` | File path of submatrix data (Prediction output) | `./data_model_out/OUTPUT/GM12878__2.0M_10Kb_KR_DFHiC_00005ep.npz` |
| `-o` | Directory path of output data | `./data_model_out` |


> *The output of iEnhance doesn't need to create a chromosome matrix; it's already done within the output file of the model prediction.*
> Output: 
> * {OutputDirectory}/{Prefix}/{InputName}.npz
