from __future__ import absolute_import

import os
import sys

import numpy as np

work_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 工作路径，用于存放代码
parent_path = os.path.abspath(os.path.join(work_path, '..')) # 上一级路径
sys.path.append(work_path) # 加载工作路径

from cann.cann import CANN_Tracker
from got10k.experiments.got10k import ExperimentGOT10k
from got10k.experiments.otb import ExperimentOTB
from utils import gen_ops

dataset = 'OTB100'
mode = 'eval'



if __name__ == '__main__':

    tracker_name = 'Test' 
    net_path = os.path.join(work_path, 'pretrained/siamfc/siamfc_alexnet_e50_download.pth')
    cann_path = os.path.join(work_path, 'pretrained/otb100_1/siamfc_cann_e49.pth')
    data_dir = gen_ops.get_data_path(dataset='OTB100', mode='eval') 

    tracker = CANN_Tracker(net_path=net_path, cann_path=None, tracker_name=tracker_name)
    
    exp_tool = gen_ops.get_exp_tool(dataset=dataset, mode=mode, data_dir=data_dir) 
    
    exp_tool.comp(tracker, name1='SiamFC', name2='Ours', is_visualize=False) # 运行测试
    # exp_tool.report([tracker.name]) # 测试结果报告
    
