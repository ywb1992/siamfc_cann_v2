from __future__ import absolute_import

import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # 加载工作路径
work_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 工作路径

from cann.cann import CANN_Tracker
from got10k.datasets import *


def setup_seed(seed):
    '''
        func: 固定随机种子
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    setup_seed(0) # 固定随机种子
    
    
    net_path = os.path.join(work_path, 'pretrained/siamfc/siamfc_alexnet_e50_download.pth')
    save_dir = os.path.join(work_path, 'pretrained')
    # data_dir = os.path.join(work_path, 'data/train/GOT10K')
    data_dir = os.path.join(work_path, 'data/eval/OTB100')
    cann_path = os.path.join(work_path, 'pretrained/2024_05_11_03_11_00/siamfc_cann_e80.pth')
    
    # seqs = GOT10k(data_dir, subset='train', return_meta=True) # 加载数据集
    seqs = OTB(data_dir, version=2015, download=False) # 加载数据集
    tracker = CANN_Tracker(net_path=net_path, mode='train') # 加载模型
    tracker.train_over(seqs, save_dir=save_dir) # 进行训练
