# Deformable 3D Convolution for Video Super-Resolution
Pytorch implementation of "[Deformable 3D Convolution for Video Super-Resolution](https://arxiv.org/pdf/2004.02803.pdf)"

Code will be released soon.

```
@Article{ying2020deformable,
  author    = {Xinyi Ying, Longguang Wang, Yingqian Wang, Weidong Sheng, Wei An, Yulan Guo},
  title     = {Deformable 3D Convolution for Video Super-Resolution},
  journal   = {arXiv preprint arXiv:2004.02803},
  year      = {2020},
}
```

## Overview
<img src="https://github.com/XinyiYing/D3Dnet/blob/master/images/Network.jpg" width="550" height="300" />

Figure 1. An illustration of deformable 3D convolution network (D3Dnet). 

<img src="https://github.com/XinyiYing/D3Dnet/blob/master/images/D3D.jpg" width="1100" height="450" />

Figure 2. Deformable 3D convolution (D3D).
## Results
### Quantitative Results
Table 1. PSNR/SSIM achieved by different methods.

<img src=https://github.com/XinyiYing/D3Dnet/blob/master/images/table1.JPG>

Table 2. [T-MOVIE and MOVIE](https://github.com/XinyiYing/MOVIE) achieved by different methods.

<img src="https://github.com/XinyiYing/D3Dnet/blob/master/images/table2.JPG" width="400" height="60" />

### Qualitative Results
<img src=https://github.com/XinyiYing/D3Dnet/blob/master/images/compare.jpg>
Figure 3. Qualitative results achieved by different methods. Blue boxes represent the temporal profiles among different frames.
