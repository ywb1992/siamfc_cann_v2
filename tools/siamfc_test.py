from __future__ import absolute_import

import os
import sys

work_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 工作路径
sys.path.append(work_path) # 加载工作路径

from got10k.experiments.otb import ExperimentOTB
from got10k.experiments.got10k import ExperimentGOT10k
from siamfc import TrackerSiamFC

if __name__ == '__main__':
    net_path = os.path.join(work_path, 'pretrained/siamfc/siamfc_alexnet_e50_download.pth')
    # failure_path = 'D:\\MyFolders\\project\\CANN\\SiamFC_CANN_v2\\failure_20_6'
    
    tracker = TrackerSiamFC(net_path=net_path, failure_path=None) # 加载模型
    data_dir = os.path.join(work_path, 'data/eval/OTB100')
    # data_dir = os.path.join(work_path, 'data/eval/GOT10k')
    
    e = ExperimentOTB(data_dir, version=2015, download=False) # 加载数据集
    # e = ExperimentGOT10k(data_dir, subset='val') # 加载数据集
    e.run(tracker, visualize=True, is_record_delta=True) # 运行测试
    e.report([tracker.name]) # 测试结果报告
    
