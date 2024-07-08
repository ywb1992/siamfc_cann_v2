"""
这个库提供了针对数值、矩阵等的操作，包括归一化、取整等。
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

from . import gen_ops, img_ops

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

def odd(x):
    '''
    Functions: 将输入的 x 变为最临近的奇数
    '''
    if isinstance(x, torch.Tensor):
        x = torch.round(x)
        return (x // 2 * 2 + 1).int()
    elif isinstance(x, np.ndarray):
        x = np.round(x)
        return (x // 2 * 2 + 1).astype(int)
    else:
        x = round(x)
        return int(x // 2 * 2 + 1)
        
def even(x):
    '''
    Functions: 将输入的 x 变为最临近的偶数
    '''
    if isinstance(x, torch.Tensor):
        x = torch.round(x)
        return (x // 2 * 2).int()
    elif isinstance(x, np.ndarray):
        x = np.round(x)
        return (x // 2 * 2).astype(int)
    else:
        x = round(x)
        return int(x // 2 * 2)

def min_max_norm(x: torch.Tensor):
    # x 为一个 3D 的 tensor，一维度是 batch，我们要对每一个矩阵分别 min-max 归一化
    batch_size= x.shape[0]
    min_val = torch.min(x.reshape(batch_size, -1), dim=1, keepdim=True).values.reshape(batch_size, 1, 1)
    max_val = torch.max(x.reshape(batch_size, -1), dim=1, keepdim=True).values.reshape(batch_size, 1, 1)
    # 应用min-max归一化
    x_min_max_normalized = (x - min_val) / (max_val - min_val)
    return x_min_max_normalized

def distribution_norm(x: torch.Tensor):
    # x 为一个 3D 的 tensor，前两个维度是 batch
    # 这里保证要 x 大于 0 才行哟
    batch_size = x.shape[0]
    sum_val = torch.sum(x, dim=[1, 2], keepdim=True)
    # 应用分布归一化
    x_distribution_normalized = x / sum_val
    return x_distribution_normalized

def roll(mat, dir, len):
    '''
    Functions: 找到了三个尺度中合适的尺度 index 后, 使其进行朝着 dir 方向进行循环位移. 得到的矩阵复制到其它尺度
    Params: index[三个尺度中, 要移动哪一个]; dir[方向]
    '''
    
    dir = torch.round(dir.squeeze().detach()).to(torch.int32)
    dir = tuple(dir.tolist())
    
    mat = torch.roll(mat, shifts=dir, dims=(0, 1))
    mat = mat.broadcast_to((1, len, len))
    return mat