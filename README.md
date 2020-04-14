# Deformable 3D Convolution for Video Super-Resolution
Pytorch implementation of deformable 3D convolution network (D3Dnet). [<a href="https://arxiv.org/pdf/2004.02803.pdf">PDF</a>] <br><br>

Our code is based on cuda and can perform deformation in any dimension of 3D convolution.

Code will be released soon.

## Architecture of D3Dnet
<img src="https://github.com/XinyiYing/D3Dnet/blob/master/images/Network.jpg" width="550" height="300" /><br>

## Architecture of D3D
<img src="https://github.com/XinyiYing/D3Dnet/blob/master/images/D3D.jpg" width="1100" height="450" /><br>


## Quantitative Results
Table 1. PSNR/SSIM achieved by different methods.

<img src=https://github.com/XinyiYing/D3Dnet/blob/master/images/table1.JPG>

Table 2. T-MOVIE and MOVIE achieved by different methods.

<img src="https://github.com/XinyiYing/D3Dnet/blob/master/images/table2.JPG" width="400" height="60" />

We have organized the Matlab code framework of Video Quality Assessment metric T-MOVIE and MOVIE. [<a href="https://github.com/XinyiYing/MOVIE">Code</a>] <br> Welcome to have a look and use our code.

## Qualitative Results
<img src=https://github.com/XinyiYing/D3Dnet/blob/master/images/compare.jpg>
Qualitative results achieved by different methods. Blue boxes represent the temporal profiles among different frames.

## Citiation
```
@article{SAM,<br>
  title={A Stereo Attention Module for Stereo Image Super-Resolution},<br>
  author={Ying, Xinyi and Wang, Yingqian and Wang, Longguang and Sheng, Weidong and An, Wei and Guo, Yulan},<br>
  journal={IEEE Signal Processing Letters},<br>
  year={2020},<br>
  publisher={IEEE}<br>
}<br>
```
## Contact
Please contact us at ***yingxinyi18@nudt.edu.cn*** for any question.

