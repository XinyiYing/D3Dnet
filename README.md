# Deformable 3D Convolution for Video Super-Resolution
Pytorch implementation of "[Deformable 3D Convolution for Video Super-Resolution](https://ieeexplore.ieee.org/document/8998204)", SPL 2020

## Overview
<img src="https://github.com/XinyiYing/D3Dnet/blob/master/images/Network.jpg" width="600" height="500" />

Figure 1. An illustration of deformable 3D convolution network (D3Dnet). (a) The overall framework. (b) The residual deformable 3D convolution (resD3D) block for simultaneous appearance and motion modeling. (c) The residual block for the reconstruction of SR results.

<img src="https://github.com/XinyiYing/D3Dnet/blob/master/images/D3D.jpg" width="600" height="500" />

Figure 2. Toy example of deformable 3D convolution (D3D).

## Results
### Quantitative Results
Table 1. PSNR/SSIM achieved by different methods.

<img src=https://github.com/XinyiYing/D3Dnet/blob/master/images/table1.jpg>

Table 2. T-MOVIE and MOVIE achieved by different methods.

<img src=https://github.com/XinyiYing/D3Dnet/blob/master/images/table2.jpg>

### Qualitative Results
<img src=https://github.com/XinyiYing/D3Dnet/blob/master/images/compare.jpg>
Figure 3. Qualitative results achieved by different methods. Blue boxes represent the temporal profiles among different frames.
