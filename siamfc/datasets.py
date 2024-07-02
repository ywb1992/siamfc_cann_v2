from __future__ import absolute_import, division

import cv2
import numpy as np
from torch.utils.data import Dataset

__all__ = ['Pair']


class Pair(Dataset):

    def __init__(self, seqs, transforms=None,
                 pairs_per_seq=1):
        super(Pair, self).__init__() # 继承 Dataset 类，同时具有 Transforms 的功能
        self.seqs = seqs
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq # 训练阶段，每个视频序列采样多少对帧（第一帧为模板），这里设置成 1
        self.indices = np.random.permutation(len(seqs)) # 打乱视频序列的索引（1-9335）
        self.return_meta = getattr(seqs, 'return_meta', False) # 有没有元信息

    def __getitem__(self, index):
        # indices 为视频序列的索引列表的打乱，这里取了余数，保证不会越界
        # index 代表的是视频序列的帧文件路径
        index = self.indices[index % len(self.indices)]

        # get filename lists and annotations
        if self.return_meta:
            # 返回的是关于一整个视频序列：[帧的绝对路径, 每帧的标注框, 视频的 meta 信息]
            img_files, anno, meta = self.seqs[index]
            # meta 信息里面的关键指标：光照度
            vis_ratios = meta.get('cover', None)
        else:
            img_files, anno = self.seqs[index][:2]
            vis_ratios = None
        
        # filter out noisy frames（不太好的帧去掉，好的帧保留；但我们这里要注意保留连续的帧，所以还得改改训练方式）
        ## 我们在训练的时候，应当选取一帧作为模板帧，然后跨越 50~100 帧，寻找间隔不超过 10 帧的图像作为起始位置和搜索区域
        val_indices = self._filter(
            cv2.imread(img_files[0], cv2.IMREAD_COLOR),
            anno, vis_ratios) # 返回这个视频里好的帧的索引
        # 如果好的帧不超过 2 帧，那就重新找
        if len(val_indices) < 2:
            index = np.random.choice(len(self))
            return self.__getitem__(index)

        # 得到好的帧序列后，随机取两帧(帧的 index 差小于100)，一帧为模板，一帧为搜寻，而且不是连续的......这样哪来的BPTT啊我靠）
        rand_z, rand_x = self._sample_pair(val_indices)

        # 读取图片张量，但是 cv2.imread 一般是读成 BGR 格式，我们需要转成 RGB 格式
        z = cv2.imread(img_files[rand_z], cv2.IMREAD_COLOR)
        x = cv2.imread(img_files[rand_x], cv2.IMREAD_COLOR)
        # 转为 RGB 形式
        z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        # 获取标注框
        box_z = anno[rand_z]
        box_x = anno[rand_x]

        # 打包成元组，并经过裁剪、缩放，最终成为样本区域 127*127，搜索区域 255*255 的图像
        item = (z, x, box_z, box_x)
        if self.transforms is not None:
            item = self.transforms(*item)
        
        # 最后返回的是：(样本图像，搜索区域)，训练时并不需要知道标注框（因为只是在 track 的时候搞）
        return item
    
    def __len__(self):
        # 训练阶段，每个视频序列采样多少对帧（第一帧为模板） * 视频序列 = 总长度
        return len(self.indices) * self.pairs_per_seq
    
    def _sample_pair(self, indices):
        n = len(indices)
        assert n > 0

        if n == 1:
            return indices[0], indices[0] 
        elif n == 2:
            return indices[0], indices[1]
        else:
            for i in range(100): # 随机找 2 帧，差小于 100 帧
                rand_z, rand_x = np.sort(
                    np.random.choice(indices, 2, replace=False))
                if rand_x - rand_z < 100:
                    break
            else:
                rand_z = np.random.choice(indices) # 如果找不到 ... 那就返回两帧一样的 ...
                rand_x = rand_z

            return rand_z, rand_x
    
    def _filter(self, img0, anno, vis_ratios=None):
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