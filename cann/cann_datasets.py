from __future__ import absolute_import, division

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from siamfc import siamfc
from utils import ops

__all__ = ['Pair']


class CANN_Pair(Dataset):
    '''
    Functions: 用于训练的数据集
    '''
    def __init__(self, siamfc:siamfc.TrackerSiamFC, 
                 seqs):
        super(CANN_Pair, self).__init__() # 继承 Dataset 类
        self.siamfc = siamfc
        self.seqs = seqs # 这个是 got10k 里面的定义
        self.indices = np.random.permutation(len(seqs)) # 打乱视频序列的索引（1-9335）
        self.return_meta = getattr(seqs, 'return_meta', False) # 有没有元信息
        
    def __getitem__(self, index):
        '''
        Functions: 返回一组训练数据
        '''
        # indices 为视频序列的索引列表的打乱，这里取了余数，保证不会越界
        # index 代表的是视频序列的帧文件路径
        index = self.indices[index % len(self.indices)]

        if self.return_meta:
            # 返回的是关于一整个视频序列：[帧的绝对路径, 每帧的标注框, 视频的 meta 信息]
            img_files, annos, meta = self.seqs[index]
            # meta 信息里面的关键指标：光照度
            vis_ratios = meta.get('cover', None)
        else:
            img_files, annos = self.seqs[index][:2]
            vis_ratios = None
        
        # filter out noisy frames（不太好的帧去掉，只保留好的帧，然后把好的帧作为第一帧）
        val_indices = self._filter(
            cv2.imread(img_files[0], cv2.IMREAD_COLOR),
            annos, vis_ratios) # 返回这个视频里好的帧的索引

        # 得到好的序列后，随机选取前一半序列里的一帧作为模板帧，然后随机一个长度（30~300），拿出剩下的帧
        seq_imgs_path, seq_annos, seq_len = self._sample_pair(val_indices, img_files, annos)
        if seq_imgs_path is None:
            index = np.random.choice(len(self))
            return self.__getitem__(index)
        
        return seq_imgs_path, seq_annos, seq_len
    
    def __len__(self):
        # 视频序列总长度
        return len(self.indices) 
    
    def _sample_pair(self, indices, img_files, annos):
        '''
        Functions: 随机寻找一个序列
        '''
        n = len(indices)

        if n <= 10:
            return None, None, None
        else: 
            # 在 indices 的前半中寻找模板帧
            # 然后固定长度 100，取剩下的图片、标注并返回
            for _ in range(50):
                seq_len = 500
                exemplar_id = np.random.choice(indices[:n//2])
                if exemplar_id + seq_len < len(img_files):
                    return  (img_files[exemplar_id : exemplar_id + seq_len],
                             annos[exemplar_id : exemplar_id + seq_len],
                             seq_len)
            else:
                return None, None, None
    
    def _filter(self, img0, anno, vis_ratios=None):
        '''
        Functions: 寻找好的帧
        '''
        size = np.array(img0.shape[1::-1])[np.newaxis, :] # 获取图片的宽高（所有帧都一样）
        areas = anno[:, 2] * anno[:, 3] # 计算每一帧标注框的面积

        # acceptance conditions，把好的数据筛选出来
        ## 不一定是连续的？？？
        ## 我在训练的时候一定要求帧连续，所以这里可能得着重筛选一下
        c1 = areas >= 20
        c2 = np.all(anno[:, 2:] >= 20, axis=1)
        c3 = np.all(anno[:, 2:] <= 500, axis=1)
        c4 = np.all((anno[:, 2:] / size) >= 0.01, axis=1)
        c5 = np.all((anno[:, 2:] / size) <= 0.5, axis=1)
        c6 = (anno[:, 2] / np.maximum(1, anno[:, 3])) >= 0.25
        c7 = (anno[:, 2] / np.maximum(1, anno[:, 3])) <= 4
        if vis_ratios is not None:
            c8 = (vis_ratios > max(1, vis_ratios.max() * 0.3))
        else:
            c8 = np.ones_like(c1)
        
        mask = np.logical_and.reduce(
            (c1, c2, c3, c4, c5, c6, c7, c8))
        val_indices = np.where(mask)[0]

        return val_indices
    
    
