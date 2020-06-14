from math import sqrt
import matplotlib.pyplot as plt
from dcn.modules.deform_conv import *
import functools
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, upscale_factor, in_channel=1, out_channel=1, nf=64):
        super(Net, self).__init__()
        self.upscale_factor = upscale_factor
        self.in_channel = in_channel

        self.input = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.residual_layer = self.make_layer(functools.partial(ResBlock_3d, nf), 5)
        self.TA = nn.Conv2d(7 * nf, nf, 1, 1, bias=True)
        ### reconstruct
        self.reconstruct = self.make_layer(functools.partial(ResBlock, nf), 6)
        ###upscale
        self.upscale = nn.Sequential(
            nn.Conv2d(nf, nf * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(nf, out_channel, 3, 1, 1, bias=False),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        b, c, n, h, w = x.size()
        residual = F.interpolate(x[:, :, n // 2, :, :], scale_factor=self.upscale_factor, mode='bilinear',
                                 align_corners=False)
        out = self.input(x)
        out = self.residual_layer(out)
        out = self.TA(out.permute(0,2,1,3,4).contiguous().view(b, -1, h, w))  # B, C, H, W
        out = self.reconstruct(out)
        ###upscale
        out = self.upscale(out)
        out = torch.add(out, residual)
        return out


class ResBlock_3d(nn.Module):
    def __init__(self, nf):
        super(ResBlock_3d, self).__init__()
        self.dcn0 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.dcn1 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x

class ResBlock(nn.Module):
    def __init__(self, nf):
        super(ResBlock, self).__init__()
        self.dcn0 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.dcn1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x


if __name__ == "__main__":
    net = Net(4).cuda()
    from thop import profile
    input = torch.randn(1, 1, 7, 320, 180).cuda()
    flops, params = profile(net, inputs=(input,))
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))


