"""
这个库提供了图片操作、数值操作外的其他一般性操作。
"""

from __future__ import absolute_import, division

import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from . import img_ops, num_ops

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')


def init_weights(model, gain=1):
    '''
    Functions: 初始化参数
    '''
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain) # Xavier 初始化方法
            if m.bias is not None:
                nn.init.constant_(m.bias, 0) #常数化初始化
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
