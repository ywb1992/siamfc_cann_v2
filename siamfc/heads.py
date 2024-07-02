from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SiamFC']


# 完成互相关操作
class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
    
    def forward(self, z, x):
        # z 为样本特征 256*6*6, x 为候选特征 256*22*22
        return self._fast_xcorr(z, x) * self.out_scale
    
    def _fast_xcorr(self, z, x):
        # fast cross correlation
        # z:(bs, 256, 6, 6); x:(bs, 256, 22, 22) 
        nz = z.size(0) # nz = bs
        nx, c, h, w = x.size() # nx, c, h, w = bs, 256, 22, 22
        x = x.view(-1, nz * c, h, w) # (1, bs * 256, 22, 22)
        
        # 以 z(bs, 256, 6, 6) 为卷积核，x(1, bs*256, 22, 22) 为特征进行分组卷积
        # 最后得到 (1, bs, 15, 15)
        out = F.conv2d(x, z, groups=nz) 
        out = out.view(nx, -1, out.size(-2), out.size(-1)) # 转化为 (bz, 1, 15, 15)
        return out
