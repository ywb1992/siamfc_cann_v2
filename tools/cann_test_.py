from __future__ import absolute_import

import os
import sys

work_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 工作路径
sys.path.append(work_path) # 加载工作路径

from cann.cann_ import CANN_Tracker
from got10k.experiments.got10k import ExperimentGOT10k
from got10k.experiments.otb import ExperimentOTB

if __name__ == '__main__':

    net_path = os.path.join(work_path, 'pretrained/siamfc/siamfc_alexnet_e50_download.pth')
    cann_path = os.path.join(work_path, 'pretrained/otb100_1/siamfc_cann_e49.pth')
    
    tracker = CANN_Tracker(net_path=net_path, cann_path=None, failure_path=None) # 加载模型
    data_dir = os.path.join(work_path, 'data/eval/OTB100')
    # data_dir = os.path.join(work_path, 'data/eval/GOT10k') 
    
    e = ExperimentOTB(data_dir, version=2015, download=False) # 加载数据集
    # e = ExperimentGOT10k(data_dir, subset='val') # 加载数据集
    e.run(tracker, visualize=True, is_record_delta=False) # 运行测试
    e.report([tracker.name]) # 测试结果报告
