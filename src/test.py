


import os
import os.path as osp
import logging

import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


imgs_dir='./data3/image'
save_dir=''
img_names = os.listdir(imgs_dir)
for i in img_names:
    img_path=imgs_dir+img_names
    img = Image.open(img_path)
    img_nd = np.array(img)
    img_nd[img_nd==2] = 1
    f_new_path= save_dir+img_names
    cv2.imwrite(f_new_path,img_nd)