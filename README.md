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




2. prepare HiC-sequencing data and make Hi-C contact map


```
bash data_download_downsample.sh https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551550/suppl/GSM1551550_HIC001_merged_nodups.txt.gz GSM1551550_HIC001 hg19 2000000 ./juicer_tools.jar KR data
```
>We download and process GM12879 cell line, which is aligned based on hg19.
>You can modify the arguments, such as the data download URL, the file name to save, the prefix of the reference genome file, 
>the read count to sample, the path to the Juicer tools, the normalization method to apply (NONE, KR, VC, etc.), resolution,
>and the output directory name, as needed, in the bash script.
>If you put these arguments in the command line, these should be placed in the order as above: 
><data_download_url>, <file_name>, <ref_genome>, <downsampled_read>, <juicertools>, <normalization>, <resolution>, <saved_in>.


>If the total number of reads in the downloaded file is less than or equal to the requested downsampled read count, 
>the downsampling step and its subsequent processes will be skipped.
>Output: 
> * Original data: ./<saved_in>/
> * Downsampled data (if applicable): ./<saved_in>_downsampled_<downsampled_read>/




3. Transform Hi-C contact map into input matrix of each model


* Input matrix for training model


```
bash data_generate_for_training.sh -i ./data -d ./data_downsampled_16 -b 10000 -m DFHiC -g ./hg19.txt -r 2000000 -o ./ -n KR -s 300 -t "1 2 3 4 5 6 7 8 9 10 11 12 13 14" -v "15 16 17" -p "18 19 20 21 22"
```
>You should specify **required arguments** as above. This Python code needs a chromosome length of reference genome **.txt** file like **hg19.txt and mm9.chrom.sizes** in the HiHiC directory. 
>Note: In the case of HiCPlus, if validation chromosome is provided, it will be automatically incorporated into the training set.

| Argument | Description | Example |
|----------|-------------|---------|
| `-i` | Hi-C data directory containing .txt files (Directory of Hi-C contact pare files) | `/HiHiC/data` |
| `-d` | Hi-C downsampled data directory containing .txt files (Directory of downsampled Hi-C contact pare files) | `/HiHiC/data_downsampled_16` |
| `-b` | Resolution (binning size; base pair length) | `10000` | 
| `-m` | Model name that you want to use (One of HiCARN, DeepHiC, HiCNN, HiCSR, DFHiC, HiCPlus, and iEnhance) | `DFHiC` |
| `-g` | Reference genome length file, your data is based on | `./hg19.txt` |
| `-r` | Number of the read count | `2000000` |
| `-o` | Parent directory path for saving output (Child directory named as the model name will be generated under this.) | `./` |
| `-s` | Max value of Hi-C matrix | `300` |
| `-n` | Normalization of Hi-C matrix | `KR` |
| `-t` | Chromosome numbers of training set | `"1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17"` |
| `-v` | Chromosome numbers of validation set | `"15 16 17"` |
| `-p` | Chromosome numbers of prediction set | `"18 19 20 21 22"` |


* Input matrix for model prediction (Enhancement with pretrained model)


```
bash data_generate_for_prediction.sh -i ./data -b 10000 -m DFHiC -g ./hg19.txt -o ./ -n KR -s 300
```
| Argument | Description | Example |
|----------|-------------|---------|
| `-i` | Hi-C data directory containing .txt files (Directory of Hi-C contact pare files) | `/HiHiC/data` |
| `-b` | Resolution (binning size; base pair length) | `10000` | 
| `-m` | Model name that you want to use (One of HiCARN, DeepHiC, HiCNN, HiCSR, DFHiC, HiCPlus, and iEnhance) | `DFHiC` |
| `-g` | Reference genome length file, your data is based on | `./hg19.txt` |
| `-o` | Parent directory path for saving output (Child directory named as the model name will be generated under this.) | `./` |
| `-s` | Max value of Hi-C matrix | `300` |
| `-n` | Normalization of Hi-C matrix | `KR` |






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
bash model_train.sh -m DFHiC -e 500 -b 16 -g 0 -o ./checkpoints_DFHiC -l ./log -t ./data_DFHiC/train -v ./data_DFHiC/valid
```
>You should specify the required arguments of the model you'd like to use, such as **model name, training epoch, batch size, GPU ID, output model directory, loss log directory, training data directory**, and **validation data directory**. 

> *All the deep learning model codes were downloaded from each author's GitHub and modified for performance comparison. For light memory storage, pre-trained weights and data have been removed*.


| Argument | Description | Example |
|----------|-------------|---------|
| `-m` | Name of the model (One of HiCARN, DeepHiC, HiCNN, HiCSR, DFHiC, HiCPlus, SRHiC, iEnhance) | `DFHiC` |   
| `-e` | Number of train epoch | `500` |
| `-b` | Number of batch size | `16` | 
| `-g` | Number of GPU ID  | `0` |
| `-o` | Directory path of output models  | `./checkpoints_DFHiC` |   
| `-l` | Directory path of training log | `./log` |
| `-t` | Directory path of input training data | `./data_DFHiC/train_KR_300` | 
| `-v` | Directory path of input validation data | `./data_DFHiC/valid_KR_300` |   






Ⅴ. HiC contact map enhancement with pretrained weights
----------------------------------------------------------


1. Change your working directory to HiHiC


```
cd /path/to/HiHiC/directory
```




2. Enhance the low resolution data you have


```
bash model_prediction.sh -m DFHiC -c ./checkpoints_DFHiC/00005_0.02.38_0.0006605307.npz -b 16 -g 0 -r 2000000 -i ./data_DFHiC/test/test_ratio16.npz -o ./output_DFHiC 
```

>You should specify the required arguments of the model you'd like to use, such as **model name, checkpoints file path, batch size, GPU ID, number of the read count, input data path, and output data directory for saving enhanced data**. When you use SRHiC, the checkpoint file need .meta format.


| Argument | Description | Example |
|----------|-------------|---------|
| `-m` | Name of the model (One of HiCARN, DeepHiC, HiCNN, HiCSR, DFHiC, HiCPlus, and iEnhance) | `DFHiC` |
| `-c` | File path of checkpoint | `./checkpoints_DFHiC/00005_0.02.38_0.0006605307.npz` |
| `-b` | Number of batch size | `16` |
| `-g` | Number of GPU ID  | `0` |
| `-r` | Number of the read count | `2000000` |
| `-i` | File path of input data | `./data_DFHiC/test/test_ratio16.npz` |
| `-o` | Directory path of output ehnhanced data | `./output_DFHiC` |

> *Without training, you can use pre-trained models in our platform. The pre-trained model weights can be downloaded via FTP.*




3. Create chromosom matrix with enhanced submatrix (Except for iEnhance)


```
python data_make_whole.py -m DFHiC -i ./output_DFHiC/DFHiC_predict_2000000_00005.npz  -o ./predicted_byDFHiC/
```

>You should specify the required arguments of the model you'd like to use, such as **model name, input submatrix, output data directory**. The output of iEnhance doesn't need to create a chromosome matrix; it's already done within the output file of the model.


| Argument | Description | Example |
|----------|-------------|---------|
| `-m` | Name of the model (One of HiCARN, DeepHiC, HiCNN, HiCSR, DFHiC, HiCPlus, and iEnhance) | `DFHiC` |
| `-i` | File path of submatrix data (output of model prediction) | `./output_DFHiC/DFHiC_predict_2000000_00005.npz` |
| `-o` | Directory path to save chromosome matrix | `./predicted_byDFHiC/` |
