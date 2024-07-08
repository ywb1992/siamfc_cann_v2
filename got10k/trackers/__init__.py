from __future__ import absolute_import

import time
from collections import namedtuple

import numpy as np
from PIL import Image

from ..utils.viz import show_frame


class Tracker(object):
    '''
    Fucntions: 一个父类, 定义了 init(追踪器初始化), update(逐帧的更新), track(整体的追踪流程) 三个方法
    '''
    def __init__(self, name, is_deterministic=False, **kwargs):
        
        self.name=name
        self.is_deterministic = is_deterministic
        self.cfg = self.parse_args(**kwargs)
    
    # init 需要子类实现
    def init(self, image, box):
        raise NotImplementedError()

    def parse_args(self, **kwargs):
        '''
        Fucntions: 初始化所有超参数
        '''
        cfg = {
            # siamfc parameters
            'out_scale': 0.001, # 得到响应图后，乘以该数值作为最终响应图
            'exemplar_sz': 127, # 样本图像输入net的大小
            'instance_sz': 255, # 搜索区域输入net的大小
            'context': 0.50,
            # cann parameters
            'len': 85, # 
            'steps': 8, # CANN 进行动力学响应的轮数
            'dt': 1, # 时间步长，默认为 1
            'tolerance_height': 0.80,
            'tolerance_dis': 64,
            'tolerant_ratio_max': 0.65, # 上限容忍度
            'tolerant_ratio_min': 0.15, # 下限容忍度
            'IoU_thresold': 0.15,
            'scale_factor': 2, 
                # trainable parameters
                'tau': 0.7133, # 时间常数
                'A': 2.0946, # 响应常数
                'k': 0.0803, # 抑制因子
                'a': 0.4040, # 空间常数
                'factor0': 1.5559, # mu 系数
                'factor1': 0.3018, # mot 系数 
                'factor2': 0.7096, # mix 系数
                'mix_factor': 0.88, # 混合 U 与 res 的系数
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 5, # 响应图上采样的倍数
            'total_stride': 8, # 步距参考 Alexnet 里面的卷积层, 2 * 2 * 2 = 8
            # train parameters
            'epoch_num': 200, # 训练总轮数
            'batch_size': 8, # 批次大小
            'num_workers': 0, # 载入数据的线程数
            'initial_lr': 1e-2, # 初始学习率（指数衰减学习率）
            'ultimate_lr': 1e-4, # 终止学习率（指数衰减学习率）
            'weight_decay': 5e-4, # SGD 权重衰减
            'momentum': 0.9, # SGD 动量
            'r_pos': 16,
            'r_neg': 0
        }
        
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)

    # update 需要子类实现
    def update(self, image):
        raise NotImplementedError()

    # track 在本项目中由子类实现
    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            image = Image.open(img_file)
            if not image.mode == 'RGB':
                image = image.convert('RGB')

            start_time = time.time()
            if f == 0:
                self.init(image, box)
            else:
                boxes[f, :] = self.update(image)
            times[f] = time.time() - start_time

            if visualize:
                show_frame(image, boxes[f, :])

        return boxes, times

from .identity_tracker import IdentityTracker
