# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:59
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : model.py
"""

"""
import torch.nn as nn

from .unet_base import *
from .nested_unet_base import *


class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()
        self.n_channels = cfg.n_channels
        self.n_classes = cfg.n_classes
        self.bilinear = cfg.bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, self.bilinear)
        self.up2 = Up(512, 256, self.bilinear)
        self.up3 = Up(256, 128, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class NestedUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_channels = cfg.n_channels
        self.n_classes = cfg.n_classes
        self.deepsupervision = cfg.deepsupervision
        self.bilinear = cfg.bilinear

        nb_filter = [32, 64, 128, 256, 512]


        self.pool = nn.MaxPool2d(2, 2)
        self.Pool2= nn.AvgPool2d(2, 2)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(self.n_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        #self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_1C = CFF2(nb_filter[0],nb_filter[1],nb_filter[0])#01
        #self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv1_1C = CFF2(nb_filter[1], nb_filter[2],nb_filter[1])#11
        #self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv2_1C = CFF2(nb_filter[2], nb_filter[3], nb_filter[2])#21
        #self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])#R5
        self.conv3_1C = CFF2(nb_filter[3], nb_filter[4], nb_filter[3])#31
        #self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_2C = CFF3(nb_filter[0], 2*nb_filter[1], nb_filter[0])#02

        #self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv1_2C = CFF3(nb_filter[1], 2*nb_filter[2], nb_filter[1])#12
        #self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])#R6
        self.conv2_2C = CFF3(nb_filter[2], 2*nb_filter[3], nb_filter[2])#22


        #self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv0_3C = CFF3(nb_filter[0], 2*nb_filter[1], nb_filter[0])#03
        #self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])#R7
        self.conv1_3C = CFF3(nb_filter[1], 2*nb_filter[2], nb_filter[1])
        #self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])#R8
        self.conv0_4C = CFF3(nb_filter[0], 2*nb_filter[1], nb_filter[0])
        if self.deepsupervision:



            self.final1 = nn.Conv2d(2*nb_filter[0], self.n_classes, kernel_size=1)

            self.final2 = nn.Conv2d(2*nb_filter[0], self.n_classes, kernel_size=1)

            self.final3 = nn.Conv2d(2*nb_filter[0], self.n_classes, kernel_size=1)

            self.final4 = nn.Conv2d(2*nb_filter[0], self.n_classes, kernel_size=1)

            self.conv1 = VGGBlock(2*nb_filter[0], nb_filter[0], nb_filter[0])
            self.conv2 = VGGBlock(2*nb_filter[0], nb_filter[0], nb_filter[0])
            self.conv3 = VGGBlock(2*nb_filter[0], nb_filter[0], nb_filter[0])
            self.conv4 = VGGBlock(2*nb_filter[0], nb_filter[0], nb_filter[0])


        else:
            self.conv1 = VGGBlock(2 * nb_filter[0], nb_filter[0], nb_filter[0])
            self.final = nn.Conv2d(nb_filter[0], self.n_classes, kernel_size=1)


    def forward(self, input):

        x0_0 = self.conv0_0(input)#L0 32
        x1_0 = self.conv1_0(self.pool(x0_0))#L1 64
        #x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x0_1 = self.conv0_1C(x0_0,x1_0) #64


        x2_0 = self.conv2_0(self.pool(x1_0))#L2 128

        #x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x1_1 = self.conv1_1C(x1_0,x2_0)#128
        #x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x0_2 = self.conv0_2C(x0_1,x1_1)# 32 64 128 32*3

        x3_0 = self.conv3_0(self.pool(x2_0))#L3
        #x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x2_1 = self.conv2_1C(x2_0,x3_0)#128*2
        #x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x1_2 = self.conv1_2C(x1_1,x2_1)#64 128 128*2 64*3
        #x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x0_3 = self.conv0_3C(x0_2,x1_2)#32 64 32*3 32*4


        x4_0 = self.conv4_0(self.pool(x3_0))#L4
        #x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))#R5
        x3_1 = self.conv3_1C(x3_0,x4_0)
        #x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))#R6
        x2_2 = self.conv2_2C(x2_1,x3_1)
        #x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))#R7
        x1_3 = self.conv1_3C(x1_2,x2_2)
        #x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))#R8
        x0_4 = self.conv0_4C(x0_3,x1_3) #32 64 32*3 32*4 32*5

        if self.deepsupervision:
            output1 =self.conv1(x0_1)
            output1 = self.final1(output1)
            output2 = self.conv2(x0_2)
            output2 = self.final2(output2)
            output3 = self.conv2(x0_3)
            output3 = self.final3(output3)
            output4 = self.conv2(x0_4)
            output4 = self.final4(output4)

            return [output1, output2, output3, output4]

        else:
            output = self.conv1(x0_4)
            output = self.final(output)
            return output
