#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from dcn.modules.deform_conv import DeformConv, _DeformConv, DeformConvPack
from dcn.modules.deform_conv import DeformConv_d, _DeformConv, DeformConvPack_d

deformable_groups = 1
B, inC, inT, inH, inW = 2, 8, 16, 16, 16
outC = 8
kT, kH, kW = 3, 3, 3
sT, sH, sW = 1, 1, 1
pT, pH, pW = 1, 1, 1

def example_dconv():
    print('============using its own offsets===========')
    input = torch.randn(B, inC, inT, inH, inW).cuda()
    dcn = DeformConvPack(inC, outC, kernel_size=[kT, kH, kW], stride=[sT, sH, sW],padding=[pT, pH, pW]).cuda()
    print('input.shape: ', input.shape)
    output = dcn(input)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print('output.shape: ', output.shape)

def example_dconv_offset():
    print('=============using extra offsets============')
    input = torch.randn(B, inC, inT, inH, inW).cuda()
    offset = torch.randn(B, kT*kH*kW*3, inT, inH, inW).cuda()
    dcn = DeformConv(inC, outC, kernel_size=[kT, kH, kW], stride=[sT, sH, sW],padding=[pT, pH, pW]).cuda()
    print('input.shape: ', input.shape)
    print('offset.shape: ', offset.shape)
    output = dcn(input, offset)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print('output.shape: ', output.shape)

def example_dconv_d():
    print('============using its own offsets===========')
    input = torch.randn(B, inC, inT, inH, inW).cuda()
    dcn = DeformConvPack_d(inC, outC, kernel_size=[kT, kH, kW], stride=[sT, sH, sW],padding=[pT, pH, pW],dimension='TW').cuda()
    #  dimension = 'T' or 'H' or 'W' or any combination of these three letters
    #  'T' represents the deformation in temporal dimension
    #  'H' represents the deformation in height dimension
    #  'W' represents the deformation in weigh dimension
    print('input.shape: ', input.shape)
    output = dcn(input)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print('output.shape: ', output.shape)

def example_dconv_offset_d():
    print('=============using extra offsets============')
    input = torch.randn(B, inC, inT, inH, inW).cuda()
    #  offset
    dimension = 'HW' # choose any dimension you want to deform
    offset = torch.randn(B, kT*kH*kW*len(dimension), inT, inH, inW).cuda()
    dcn = DeformConv_d(inC, outC, kernel_size=[kT, kH, kW], stride=[sT, sH, sW],padding=[pT, pH, pW],dimension=dimension).cuda()
    #  dimension = 'T' or 'H' or 'W' or any combination of these three letters
    #  'T' represents the deformation in temporal dimension
    #  'H' represents the deformation in height dimension
    #  'W' represents the deformation in weigh dimension
    print('input.shape: ', input.shape)
    print('offset.shape: ', offset.shape)
    output = dcn(input, offset)
    targert = output.new(*output.size())
    targert.data.uniform_(-0.01, 0.01)
    error = (targert - output).mean()
    error.backward()
    print('output.shape: ', output.shape)

if __name__ == '__main__':
    print('==================Deformable 3D convolution==================', '\n')
    # D3D deform in three dimensions
    print('=============D3D deform in three dimensions===========')
    example_dconv() # DCN using its own offsets
    example_dconv_offset() # DCN using extra offsets
    print('\n')

    # D3D available for deformable dimension
    print('============option for deformable dimension===========')
    example_dconv_d()  # DCN using its own offsets
    example_dconv_offset_d()  # DCN using extra offsets

