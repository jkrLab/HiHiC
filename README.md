HiHiC
=====
*A highlighting platform comparing deep learning models for Hi-C contact map enhancement*   
*HiHiC docker repository: [https://hub.docker.com/repositories/jkrlab](https://hub.docker.com/repositories/jkrlab)*


Ⅰ. Environment for data generation
------------------------------------
1. Clone HiHiC repository
```
git clone HiHiC
```
2. Run the docker image for data processing
```
docker run --rm --gpus all -it --name hihic_preprocess -v ${PWD}:${PWD} jkrlab/hihic_preprocess
```
3. Make symbolic link Juicer tools in the workspace of docker to HiHiC directory
```
ln -s /workspace/juicer_tools.jar /path/to/HiHiC/directory
```

  
Ⅱ. Data generation for each deep learning models
-------------------------------------------------
1. Change your working directory to HiHiC
```
cd /path/to/HiHiC/directory
```
2. Prepare HiC-sequencing read data and randomly sample reads for low resolution data
>We download and process GM12879 cell line, which is aligned based on hg19.
>You can modify options, **data download url, file name, reference genome, downsampling ratio**, and **path of juicer tools** in bash script, as you need.
>If you put this argments in command line, these should be put in order.
```
bash data_download_downsample.sh https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM1551nnn/GSM1551550/suppl/GSM1551550_HIC001_merged_nodups.txt.gz GSM1551550_HIC001 hg19 16 ./juicer_tools.jar
```
3. Generate input data of each deep leaning models
>This python code needs chromosome length .txt file of reference genome like **hg19.txt** in HiHiC directory. You also should specify **required argments** as below.
>```
>-i : Hi-C data directory containing .txt files (Directory of Hi-C contact pare files) - (example) /HiHiC/data   
>-d : Hi-C downsampled data directory containing .txt files (Directory of downsampled Hi-C contact pare files) - (example) /HiHiC/data_downsampled_16   
>-m : Model name that you want to use (One of HiCARN, DeepHiC, HiCNN2, HiCSR, DFHiC, hicplus, and SRHiC) - (example) DFHiC   
>-g : Reference genome length file, your data is based on - (example) ./hg19.txt  
>-r : Downsampling ratio of your downsampled data - (example) 16
>-o : Parent directory path for saving output (Child directory named as the model name will be generated under this.) - (example) ./
>```
```
python data_generate.py -i ./data -d ./data_downsampled_16 -m DFHiC -g ./hg19.txt -r 16 -o ./
```

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
>The model codes were downloaded from github of the each author. For lightening storage, pretrained weights were removed. You also should specified required argments of the model you want.
```
```

Ⅴ. HiC contact map enhancement
--------------------------------------
> Without training, you can use pretrained models in our platform. The pretrained model weights are can be downloaded by transfer protocol.
```

```
