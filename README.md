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

  
Ⅱ. Data generation for each deep learning models
-------------------------------------------------
1. Change your working directory to HiHiC
```
cd ../../HiHiC
```
2. Prepare HiC-sequencing read data and randomly sample reads for low resolution data
>We download and process GM12879 cell line, which is based on hg19.   
>You can modify options, **download_url, file_name, chromosome_length**, and **downsample_ratio** in bash script, as you need.
```
./data_download_downsample.sh
```

3. Generate input data of each deep leaning models
>This python code needs chromosome length file like **hg19.txt** in the same directory.
```
python data_generate.py
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


Ⅴ. Use to HiC contact map enhancement
--------------------------------------
> Without training and test, you can use pretrained models in our platform.
