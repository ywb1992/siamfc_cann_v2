from __future__ import absolute_import

import os
import sys

work_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 工作路径
sys.path.append(work_path) # 加载工作路径

from got10k.datasets import *
from siamfc import TrackerSiamFC

if __name__ == '__main__':
    data_dir = os.path.join(work_path, 'data/train/GOT10K')
    seqs = GOT10k(data_dir, subset='train', return_meta=True)
    
    net_path = os.path.join(work_path, 'pretrained/siamfc/siamfc_alexnet_e50_download.pth')
    
    tracker = TrackerSiamFC(net_path=net_path)
    tracker.train_over(seqs)
