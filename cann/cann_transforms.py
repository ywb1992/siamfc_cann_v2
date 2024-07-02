from __future__ import absolute_import, division

import numbers

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from utils import ops

__all__ = ['SiamFC_CANN_Transforms']



# class Img2Tensor(object):

#     def __call__(self, img):
#         return torch.from_numpy(img).float().permute((2, 0, 1))


# class CANNTransforms(object):
#     '''
#     Functions: 调用时
#     '''
#     def __init__(self, exemplar_sz=127, instance_sz=255, context=0.5):
#         self.exemplar_sz = exemplar_sz # 样本图像大小
#         self.instance_sz = instance_sz # 搜索区域大小
#         self.context = context # 扩张比例
#         self.img2tensor = Img2Tensor()
    
    
#     def __call__(self, exemplar_path, seq_imgs_path, seq_annos, seq_len):
#         # 先对第一帧图像进行处理
#         exemplar = ops.read_image(exemplar_path)
#         exemplar_box = seq_annos[0]
#         exemplar_box = np.array([
#             exemplar_box[1] - 1 + (exemplar_box[3] - 1) / 2,
#             exemplar_box[0] - 1 + (exemplar_box[2] - 1) / 2,
#             exemplar_box[3], exemplar_box[2]], dtype=np.float32)
#         exemplar_center, exemplar_sz = exemplar_box[:2], exemplar_box[2:]
#         # 然后把样本图像裁剪并缩放成 (127, 127) 作为模板区域
#         exemplar_sz = np.sqrt(np.prod(exemplar_sz + self.context * np.sum(exemplar_sz)))
#         exemplar_avg_color = np.mean(exemplar, axis=(0, 1))
#         exemplar = ops.crop_and_resize(exemplar, exemplar_center, exemplar_sz,
#                                             self.exemplar_sz, border_value=exemplar_avg_color)
        
#         return exemplar, seq_imgs_path, seq_annos, seq_len
    
