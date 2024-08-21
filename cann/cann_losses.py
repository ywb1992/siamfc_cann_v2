from __future__ import absolute_import, division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CenterDistanceLoss(nn.Module):
    '''
    Functions: 定义了损失函数, 包括寻找循环重心和计算中心误差(CE)两个部分
    '''
    def __init__(self, temperature=0.2):
        super(CenterDistanceLoss, self).__init__()
        self.tau = torch.tensor([temperature], device=device)
        
    def get_circular_weight_core(self, A: torch.Tensor):
        # A: (batch_size, h, w); return: (batch_size, 2) 
        batch_size = A.shape[0]
        nx, ny = A.shape[1], A.shape[2]
        mx, my = torch.meshgrid(torch.arange(nx), torch.arange(ny))
        mx, my = mx / nx * (2 * torch.pi), my / ny * (2 * torch.pi)
        mx, my = mx.broadcast_to(A.shape), my.broadcast_to(A.shape) # (batch_size, h, w)
        mx, my = mx.to(A.device), my.to(A.device)
        
        # from line to circle
        encode_x = torch.stack([torch.cos(mx), torch.sin(mx)], dim=-1) # (batch_size, h, w, 2)
        encode_y = torch.stack([torch.cos(my), torch.sin(my)], dim=-1) # (batch_size, h, w, 2)
        
        # get the center in the circle
        encode_Ax = torch.sum(encode_x * A.unsqueeze(-1), dim=[1, 2]) / \
            torch.sum(A, dim=[1, 2]).unsqueeze(-1) # (batch_size, 2)
        encode_Ay = torch.sum(encode_y * A.unsqueeze(-1), dim=[1, 2]) / \
            torch.sum(A, dim=[1, 2]).unsqueeze(-1) # (batch_size, 2)
        
        # from circle to line
        decode_Ax = torch.atan2(encode_Ax[:, 1], encode_Ax[:, 0]) # (batch_size)
        decode_Ay = torch.atan2(encode_Ay[:, 1], encode_Ay[:, 0]) # (batch_size)
        
        # larger than 0
        decode_Ax = torch.where(decode_Ax < 0, decode_Ax + 2 * torch.pi, decode_Ax) # (batch_size)
        decode_Ay = torch.where(decode_Ay < 0, decode_Ay + 2 * torch.pi, decode_Ay) # (batch_size)
        
        # from line to float-index
        decode_Ax = decode_Ax / (2 * torch.pi) * nx # (batch_size)
        decode_Ay = decode_Ay / (2 * torch.pi) * ny # (batch_size)
        
        # return circular weight center
        return torch.stack([decode_Ax, decode_Ay], dim=1)
    
    def get_center(self, A):
        # A: (batch_size, h, w); return: (batch_size, 2)
        A = torch.exp(A / self.tau) / torch.sum(torch.exp(A / self.tau), dim=(1, 2), keepdim=True)
        A = self.get_circular_weight_core(A)
        return A
        
    def forward(self, center1, center2):
        if center1.shape[0] != 1:
            center1 = center1.unsqueeze(0)
            center2 = center2.unsqueeze(0)
        distance = torch.norm(center1 - center2, dim=1) # Calculate the square of Euclidean distance
        return torch.mean(distance)  # Square the distance and return Batch-mean
