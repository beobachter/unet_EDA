# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:55
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : dataset.py
"""

"""
import os
import os.path as osp
import logging

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir   = imgs_dir
        self.masks_dir  = masks_dir
        self.scale      = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.img_names = os.listdir(imgs_dir)
        logging.info(f'Creating dataset with {len(self.img_names)} examples')

    def __len__(self):
        return len(self.img_names)

    @classmethod
    def preprocess(cls, pil_img, scale,flag):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if flag == 0:
            # mask target image
            img_nd[img_nd==3] = 3 
            img_nd[img_nd!=3] = 0
            img_nd[img_nd==3] = 1
             
            img_nd = np.expand_dims(img_nd, axis=2)
            img_nd = img_nd.transpose((2, 0, 1))
            img_trans =torch.from_numpy(img_nd)
        else:
            # grayscale input image
            # scale between 0 and 1
            img_nd = img_nd / 255
            img_nd = np.expand_dims(img_nd, axis=2)
            transform = transforms.Compose([
                #transforms.Resize(32),
                #transforms.CenterCrop(32),
                transforms.ToTensor(),

                transforms.Normalize(mean=[0.5], std=[0.5])#image=(image-mean)/std
            ])
            img_trans = transform(img_nd)
        # HWC to CHW

        return img_trans#.astype(float)

    def __getitem__(self, i):
        img_name = self.img_names[i]
        maskname = os.path.splitext(img_name)[0] + ".png"
        img_path = osp.join(self.imgs_dir, img_name)
        mask_path = osp.join(self.masks_dir, maskname)

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        assert img.size == mask.size, \
            f'Image and mask {img_name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale,1)
        mask = self.preprocess(mask, self.scale,0)

        return {'image': img, 'mask': mask}
