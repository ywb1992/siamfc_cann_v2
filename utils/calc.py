from __future__ import absolute_import, division

import cv2
import numpy as np
import torch
import torch.nn as nn

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