# Deformable 3D Convolution for Video Super-Resolution
Pytorch implementation of deformable 3D convolution network (D3Dnet). [<a href="https://arxiv.org/pdf/2004.02803.pdf">PDF</a>] <br><br>

Our code is based on cuda and can perform deformation in any dimension of 3D convolution.

## Overview

### Architecture of D3Dnet
<img src="https://raw.github.com/XinyiYing/D3Dnet/master/images/Network.jpg" width="512"/><br>

### Architecture of D3D
<img src="https://raw.github.com/XinyiYing/D3Dnet/master/images/D3D.jpg" width="1024"/><br>

## Requirements
- Python 3
- pytorch (1.0.0), torchvision (0.2.2) / pytorch (1.2.0), torchvision (0.4.0) 
- numpy, PIL
- Visual Studio 2015

## Build
***Compile deformable 3D convolution***: <br>
1. Cd to ```code/dcn```.
2. For Windows users, run  ```cmd make.bat```. For Linux users, run ```bash make.sh```. The scripts will build D3D automatically and create some folders.
3. We offer customized settings for any dimension (e.g., Temporal, Height, Width) you want to deform. See ```code/dcn/test.py``` for more details.

## Datasets

### Training dataset

1. Download the [Vimeo](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip) dataset and put the images in `code/data/Vimeo`.  
2. Cd to `code/data/Vimeo` and run `generate_LR_Vimeo90K.m` to generate training data as below:
```
  Vimeo
    └── sequences
           ├── 00001
           ├── 00002
           ├── ...
    └── LR_x4
           ├── 00001
           ├── 00002
           ├── ...		
    ├── readme.txt 
    ├── sep_trainlist.txt
    ├── sep_testlist.txt
    └── generate_LR_Vimeo90K.m      
```

### Test dataset

1. Download the dataset Vid4 and SPMC-11 dataset in https://pan.baidu.com/s/1PKZeTo8HVklHU5Pe26qUtw (Code: 4l5r) and put the folder in `code/data`.
2. (optional) You can also download Vid4 and SPMC-11 or other video datasets and prepare test data in `code/data` as below:
```
 data
  └── dataset_1
         └── scene_1
               └── hr    
                  ├── hr_01.png  
                  ├── hr_02.png  
                  ├── ...
                  └── hr_M.png    
               └── lr_x4
                  ├── lr_01.png  
                  ├── lr_02.png  
                  ├── ...
                  └── lr_M.png   
         ├── ...		  
         └── scene_M
  ├── ...    
  └── dataset_N      
```
## Results

### Quantitative Results

<img src="https://raw.github.com/XinyiYing/D3Dnet/master/images/table1.JPG" width="1024" />

<img src="https://raw.github.com/XinyiYing/D3Dnet/master/images/table2.JPG" width="550"/>

We have organized the Matlab code framework of Video Quality Assessment metric T-MOVIE and MOVIE. [<a href="https://github.com/XinyiYing/MOVIE">Code</a>] <br> Welcome to have a look and use our code.

### Qualitative Results
<img src=https://raw.github.com/XinyiYing/D3Dnet/master/images/compare.jpg>

A demo video is available at https://wyqdatabase.s3-us-west-1.amazonaws.com/D3Dnet.mp4

## Citiation
```
@article{D3Dnet,
  author = {Ying, Xinyi and Wang, Longguang and Wang, Yingqian and Sheng, Weidong and An, Wei and Guo, Yulan},
  title = {Deformable 3D Convolution for Video Super-Resolution},
  journal = {IEEE Signal Processing Letters},
  volume = {27},
  pages = {1500-1504}，
  year = {2020},
}
```

## Acknowledgement
This code is built on [[DCNv2]](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0) and [[SOF-VSR]](https://github.com/LongguangWang/SOF-VSR). We thank the authors for sharing their codes.

## Contact
Please contact us at ***yingxinyi18@nudt.edu.cn*** for any question.

