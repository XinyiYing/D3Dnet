#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _triple

from dcn.functions.deform_conv_func_seperate import DeformConvFunction


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
        super(DeformConv, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset):
        assert 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] == \
               offset.shape[1]
        return DeformConvFunction.apply(input, offset,
                                        self.weight,
                                        self.bias,
                                        self.stride,
                                        self.padding,
                                        self.dilation,
                                        self.groups,
                                        self.deformable_groups,
                                        self.im2col_step)


_DeformConv = DeformConvFunction.apply


class DeformConvPack_s(DeformConv):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(DeformConvPack_s, self).__init__(in_channels, out_channels,
                                               kernel_size, stride, padding, dilation, groups, deformable_groups,
                                               im2col_step, bias)

        out_channels = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.conv_offset = nn.Conv3d(self.in_channels,
                                     out_channels,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input):
        offset = self.conv_offset(input)
        # b,c,t,h,w = offset.shape
        # offset1 = torch.zeros(b,c,t,h,w).cuda()
        for i in range(9):
            # offset1[:, i * 3 + 1, :, :, :] = offset[:, i * 3 + 1, :, :, :]
            # offset1[:, i * 3 + 2, :, :, :] = offset[:, i * 3 + 2, :, :, :]
            offset[:, i * 3, :, :, :] = 0
        return DeformConvFunction.apply(input, offset,
                                           self.weight,
                                           self.bias,
                                           self.stride,
                                           self.padding,
                                           self.dilation,
                                           self.groups,
                                           self.deformable_groups,
                                           self.im2col_step)


class DeformConvPack_t(DeformConv):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(DeformConvPack_t, self).__init__(in_channels, out_channels,
                                               kernel_size, stride, padding, dilation, groups, deformable_groups,
                                               im2col_step, bias)

        out_channels = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        self.conv_offset = nn.Conv3d(self.in_channels,
                                     out_channels,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input):
        offset = self.conv_offset(input)
        b, c, t, h, w = offset.shape
        # offset1 = torch.zeros(b,c,t,h,w).cuda()
        for i in range(3):
            offset[:, i * 3 + 1, :, :, :] = 0
            offset[:, i * 3 + 2, :, :, :] = 0
            # offset1[:, i * 3, :, :, :] = offset[:, i * 3, :, :, :]

        return DeformConvFunction.apply(input, offset,
                                           self.weight,
                                           self.bias,
                                           self.stride,
                                           self.padding,
                                           self.dilation,
                                           self.groups,
                                           self.deformable_groups,
                                           self.im2col_step)