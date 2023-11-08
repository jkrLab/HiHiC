HiHiC
=====
*A highlighting platform comparing deep learning models for Hi-C contact map enhancement*   
*HiHiC docker repository: [https://hub.docker.com/repositories/jkrlab](https://hub.docker.com/repositories/jkrlab)*


Ⅰ. Environment for data generation
------------------------------------
1. Clone HiHiC repository
```
git clone jkrLab/HiHiC
```
2. Run the docker image for data processing
```
docker run --rm --gpus all -it --name hihic_preprocess -v ${PWD}:${PWD} jkrlab/hihic_preprocess
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
2. Prepare HiC-sequencing data and randomly sample for making low-resolution data
```
bash data_download_downsample.sh https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551550/suppl/GSM1551550_HIC001_merged_nodups.txt.gz GSM1551550_HIC001 hg19 16 ./juicer_tools.jar
```
>We download and process GM12879 cell line, which is aligned based on hg19.
>You can modify arguments, **data download URL, file name, reference genome, downsampling ratio**, and **path of juicer tools** in the bash script, as you need. 
>If you put these arguments in the command line, these should be placed in the order as above: data download URL, saving file name, reference genome, downsampling ratio, path of juicer tools
3. Generate input data for each deep learning model
```
python data_generate.py -i ./data -d ./data_downsampled_16 -m DFHiC -g ./hg19.txt -r 16 -o ./
```
>You should specify **required arguments** as above. This Python code needs a chromosome length of reference genome .txt file like **hg19.txt** in the HiHiC directory. 
>```
>-i : Hi-C data directory containing .txt files (Directory of Hi-C contact pare files) - (example) /HiHiC/data   
>-d : Hi-C downsampled data directory containing .txt files (Directory of downsampled Hi-C contact pare files) - (example) /HiHiC/data_downsampled_16   
>-m : Model name that you want to use (One of HiCARN, DeepHiC, HiCNN2, HiCSR, DFHiC, hicplus, and SRHiC) - (example) DFHiC   
>-g : Reference genome length file, your data is based on - (example) ./hg19.txt  
>-r : Downsampling ratio of your downsampled data - (example) 16
>-o : Parent directory path for saving output (Child directory named as the model name will be generated under this.) - (example) ./
>```


Ⅲ. Environment for each deep learning model
--------------------------------------------
* hicplus, HiCNN, deepHiC, or HiCARN:
```
docker run --rm --gpus all -it --name hihic_torch -v ${PWD}:${PWD} jkrlab/hihic_torch
```
* SRHiC or DFHiC:
```
docker run --rm --gpus all -it --name hihic_tensorflow -v ${PWD}:${PWD} jkrlab/hihic_tensorflow
```

Ⅳ. Training and Test
--------------------- 
```
bash model_train.sh DFHiC 500 0 /HiHiC /HiHiC /HiHiC/data_DFHiC/train /HiHiC/data_DFHiC/valid
```
> You should specify the required arguments of the model you'd like to use, such as **model name, training epoch, GPU ID, output model directory, loss log directory, training data directory**, and **validation data directory**. When you use hicplus, the validation data directory is not required.
> If you put these arguments in the command line, these should be placed in the following order: model name, training epoch, GPU ID, output model directory, loss log directory, training data directory, and validation data directory.   
> *All model codes were downloaded from each author's GitHub and modified for performance comparison. For light memory storage, pre-trained weights and data have been removed*.

Ⅴ. HiC contact map enhancement
--------------------------------------
> Without training, you can use pre-trained models in our platform. The pre-trained model weights can be downloaded by transfer protocol.
```

```
