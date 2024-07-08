from __future__ import absolute_import

import os
import sys

import numpy as np
import torch

work_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 工作路径，用于存放代码
parent_path = os.path.abspath(os.path.join(work_path, '..')) # 上一级路径
sys.path.append(work_path) # 加载工作路径

from cann.cann import CANN_Tracker
from got10k.datasets import OTB, got10k
from utils import gen_ops

dataset = 'OTB100'
mode = 'train'

if __name__ == '__main__':
    gen_ops.setup_seed(0) # 固定随机种子
    
    net_path = os.path.join(work_path, 'pretrained/siamfc/siamfc_alexnet_e50_download.pth')
    save_dir = os.path.join(work_path, 'pretrained')
    data_dir = gen_ops.get_data_path(dataset=dataset, mode=mode) 
    cann_path = os.path.join(work_path, 'pretrained/2024_05_11_03_11_00/siamfc_cann_e80.pth')
    
    seqs = gen_ops.get_train_tool(dataset=dataset, mode=mode, data_dir=data_dir) # 加载数据集
    tracker = CANN_Tracker(net_path=net_path, mode=mode) # 加载模型
    tracker.train_over(seqs, save_dir=save_dir) # 进行训练
