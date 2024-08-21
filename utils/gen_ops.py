"""
这个库提供了图片操作、数值操作外的其他一般性操作。
"""

from __future__ import absolute_import, division

import os
import platform
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from got10k.datasets import OTB, got10k
from got10k.experiments.got10k import ExperimentGOT10k
from got10k.experiments.otb import ExperimentOTB

from . import img_ops, num_ops

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
work_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 工作路径，用于存放代码
parent_path = os.path.abspath(os.path.join(work_path, '..')) # 上一级路径
sys.path.append(work_path) # 加载工作路径



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

def setup_seed(seed):
    '''
        func: 固定随机种子
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_os():
    '''
    Functions: 获得当前系统
    '''
    if platform.system() == 'Windows':
        return 'w'
    elif platform.system() == 'Linux':
        return 'l'
    else:
        return 'unknown'  # 可以增加一个默认返回值，处理其他操作系统的情况

def get_formatted_date():
    '''
    Functions: 获取当前时间的字符串形式
    '''
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S") # 获取当前时间，用于保存模型

def get_data_path(dataset='OTB100', mode='test'):
    '''
    Functions: 获取数据路径
    '''
    now_os = get_os()
    if now_os == 'w':
        if dataset == 'OTB100':
            data_path = os.path.join(parent_path, 'data/' + 'eval' + '/' + dataset)        
        else:
            data_path = os.path.join(parent_path, 'data/' + mode + '/' + dataset)
    elif now_os == 'l':
        data_path = None
    else:
        data_path = 'unknown'
    return data_path

def get_exp_tool(dataset='OTB100', mode='eval', data_dir=None):
    '''
    Functions: 获取数据集的评估工具
    '''
    if dataset == 'OTB100':
        return ExperimentOTB(data_dir, version=2015, download=False) # 加载数据集
    elif dataset == 'GOT10k':
        return ExperimentGOT10k(data_dir, subset='val') # 加载数据集
        
def get_train_tool(dataset='OTB100', mode='train', data_dir=None):
    '''
    Functions: 获取数据集的训练工具
    '''
    if dataset == 'OTB100':
        return OTB(data_dir, version=2015, download=False)
    elif dataset == 'GOT10K':
        return got10k(data_dir, subset='train')