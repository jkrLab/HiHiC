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
```
docker run --rm --gpus all -it --name hihic_preprocess -v ${PWD}:${PWD} jkrlab/hihic_preprocess
```
3. Make symbolic linc Juicer tools in the workspace of docker to HiHiC directory
```
ln -s /workspace/juicer_tools.jar /HiHiC-main
```

  
Ⅱ. Data generation for each deep learning models
-------------------------------------------------
1. Change your working directory to HiHiC
```
cd /HiHiC-main
```
2. Prepare HiC-sequencing read data and randomly sample reads for low resolution data
>We download and process GM12879 cell line, which is based on hg19.   
>You can modify options, **download_url, file_name, chromosome_length**, and **downsample_ratio** in bash script, as you need.
```
./data_download_downsample.sh
```

3. Generate input data of each deep leaning models
>This python code needs chromosome length file like **hg19.txt** in the same directory. You also should specify required arguments as below.
>```
>-i : Hi-C data directory containing .txt files (directory of Hi-C contact pare files) - example) /HiHiC-main/data   
>-d : Hi-C downsampled data directory containing .txt files (directory of downsampled Hi-C contact pare files) - example) /HiHiC-main/data_downsampled_16   
>-m : Model name that you want to use (One of HiCARN, DeepHiC, HiCNN2, HiCSR, DFHiC, hicplus, and SRHiC) - example) DFHiC   
>-g : Reference genome length file, your data is based on - example) hg19.txt'  
>-r : Downsampling ratio of your downsampled data - example) 16
>-o : Parent directory path for saving output (child directory named as the model name will be generated under this)
>```
```
python data_generate.py -i ./data -d ./data_downsampled_16 -m DFHiC -g ./hg19.txt -r 16 -o ./
```

Ⅲ. Environment for each deep learning model
--------------------------------------------
* hicplus, HiCNN, SRHiC, deepHiC, or HiCARN:
```
docker run --rm --gpus all -it --name hihic_torch -v ${PWD}:${PWD} jkrlab/hihic_torch
```
* DFHiC:
```
docker run --rm --gpus all -it --name hihic_tensorflow -v ${PWD}:${PWD} jkrlab/hihic_tensorflow
```

Ⅳ. Training and Test
---------------------


Ⅴ. HiC contact map enhancement
--------------------------------------
> Without training, you can use pretrained models in our platform.
