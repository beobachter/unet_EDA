# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:58
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : nested_unet_base.py
"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out


class CFF2(nn.Module):
    def __init__(self, F2_channels, F1_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(CFF2, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(F1_channels, out_channels, 3, padding=2,dilation=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(F2_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self,F2,F1):

        F1 = self.up(F1)
        F1 = self.conv1(F1)
        F1 = self.bn1(F1)

        F2 = self.conv2(F2)
        F2 = self.bn2(F2)

        out =  torch.cat([F1,F2],1)
        out = self.act_func(out)

        return out



class CFF3(nn.Module):
    def __init__(self,  F2_channels, F1_channels, out_channels, act_func=nn.ReLU(inplace=True)):
            super(CFF3, self).__init__()
            self.act_func = act_func
            self.conv1 = nn.Conv2d(F1_channels, out_channels, 3, padding=2,dilation=2)
            self.bn1 = nn.BatchNorm2d(out_channels)
            # self.conv2 = nn.Conv2d(F2_channels, out_channels, 1)
            # self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv3 = nn.Conv2d(F2_channels*2, out_channels, 1)
            self.bn3 = nn.BatchNorm2d(out_channels)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self,F3, F1):
            F1 = self.up(F1)
            F1 = self.conv1(F1)
            F1 = self.bn1(F1)



            # F2 = self.conv2(F2)
            # F2 = self.bn2(F2)

            F3 = self.conv3(F3)
            F3 = self.bn3(F3)

            out = torch.cat([F1, F3], 1)
            out = self.act_func(out)
            return out

# class CFF4(nn.Module):
#     def __init__(self,  F2_channels, F1_channels, out_channels, act_func=nn.ReLU(inplace=True)):
#             super(CFF4, self).__init__()
#             self.act_func = act_func
#             self.conv1 = nn.Conv2d(F1_channels, out_channels, 3, padding=2, dilation=2)
#             self.bn1 = nn.BatchNorm2d(out_channels)
#             # self.conv2 = nn.Conv2d(F2_channels, out_channels, 1)
#             # self.bn2 = nn.BatchNorm2d(out_channels)
#             # self.conv3 = nn.Conv2d(F2_channels*2, out_channels, 1)
#             # self.bn3 = nn.BatchNorm2d(out_channels)
#             self.conv4 = nn.Conv2d(F2_channels * 2, out_channels, 1)
#             self.bn4 = nn.BatchNorm2d(out_channels)
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#
#     def forward(self, F2, F3, F4, F1):
#             F1 = self.up(F1)
#             F1 = self.conv1(F1)
#             F1 = self.bn1(F1)
#
#             # F2 = self.conv2(F2)
#             # F2 = self.bn2(F2)
#             #
#             # F3 = self.conv3(F3)
#             # F3 = self.bn3(F3)
#
#             F4 = self.conv4(F4)
#             F4 = self.bn4(F4)
#
#             out = torch.cat([F1, F4], 1)
#             out = self.act_func(out)
#
#             return out
#
# class CFF5(nn.Module):#32 128 32
#     def __init__(self,  F2_channels, F1_channels, out_channels, act_func=nn.ReLU(inplace=True)):
#             super(CFF5, self).__init__()
#             self.act_func = act_func
#             self.conv1 = nn.Conv2d(F1_channels, out_channels, 3, padding=2, dilation=2)  #128_32
#             self.bn1 = nn.BatchNorm2d(out_channels)
#             # self.conv2 = nn.Conv2d(F2_channels, out_channels, 1)
#             # self.bn2 = nn.BatchNorm2d(out_channels)
#             # self.conv3 = nn.Conv2d(F2_channels*2, out_channels, 1)
#             # self.bn3 = nn.BatchNorm2d(out_channels)
#             # self.conv4 = nn.Conv2d(F2_channels * 3, out_channels, 1)
#             # self.bn4 = nn.BatchNorm2d(out_channels)
#             self.conv5 = nn.Conv2d(F2_channels * 2, out_channels, 1)  #64_32
#             self.bn5 = nn.BatchNorm2d(out_channels)
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#
#     def forward(self, F2, F3, F4 ,F5, F1):
#             F1 = self.up(F1)
#             F1 = self.conv1(F1)
#             F1 = self.bn1(F1)
#             #
#             # F2 = self.conv2(F2)
#             # F2 = self.bn2(F2)
#             #
#             # F3 = self.conv3(F3)
#             # F3 = self.bn3(F3)
#             #
#             # F4 = self.conv4(F4)
#             # F4 = self.bn4(F4)
#
#             F5 = self.conv5(F5)
#             F5 = self.bn5(F5)
#
#             out = torch.cat([F1, F5], 1)
#             out = self.act_func(out) #64
#
#             return out