from __future__ import absolute_import, division

import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from utils import num_ops

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

def crop_and_resize_optimized(img, center, size, out_size,
                              border_type=cv2.BORDER_CONSTANT,
                              border_value=(0, 0, 0),
                              interp=cv2.INTER_LINEAR):
    '''
    函数：优化的图像裁剪并缩放功能。
    先裁剪，如果裁剪区域部分在图像外，则对裁剪后的图像进行填充。
    '''
    size = round(size)

    # 计算裁剪区域的左上角和右下角坐标
    top_left = np.array([np.round(center[0] - (size - 1) / 2), 
                         np.round(center[1] - (size - 1) / 2)]).astype(int)
    bottom_right = top_left + size

    # 初始化裁剪区域在图像内的部分
    top_left_clamped = np.clip(top_left, 0, [img.shape[0], img.shape[1]])
    bottom_right_clamped = np.clip(bottom_right, 0, [img.shape[0], img.shape[1]])

    # 裁剪图像
    patch = img[top_left_clamped[0]:bottom_right_clamped[0], top_left_clamped[1]:bottom_right_clamped[1]]

    # 计算需要填充的量
    padding = [
        top_left_clamped[0] - top_left[0],  # 上侧填充
        top_left_clamped[1] - top_left[1],  # 左侧填充
        bottom_right[0] - bottom_right_clamped[0],  # 下侧填充
        bottom_right[1] - bottom_right_clamped[1]   # 右侧填充
    ]

    # 应用填充
    if np.any(padding):
        patch = cv2.copyMakeBorder(patch, padding[0], padding[2], padding[1], padding[3],
                                   border_type, value=border_value)

    # 缩放到目标尺寸
    patch = cv2.resize(patch, (out_size, out_size), interpolation=interp)

    return patch

def get_frame_difference(img1, img2):
    '''
    Functions: 转化为灰度图, 并计算差分
    '''
    img1, img2 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    img1, img2 = img1 / 255.0, img2 / 255.0
    diff = np.abs(img1 - img2)
    return diff

def get_frame_difference_torch(img1, img2):
    '''
    Functions: 转化为灰度图, 并计算差分
    '''
    img1, img2 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    img1, img2 = img1 / 255.0, img2 / 255.0
    diff = img2 - img1
    return diff

def get_cann_inputs_optimized(pre_img, img, response, sz, center, cann_len):
    '''
    Functions: 返回响应图, 运动图, 混合图, 以及取整后的尺寸(均为 (1, 271, 271))
    '''
    center, sz = np.round(center), num_ops.odd(sz) # 进行取整操作; odd 是保证为奇数
    
    # 找到图中的对应区域，并裁剪下来
    ## 首先计算响应图映射回原图像后的边界：(ly, lx, ry, rx)
    response_corners = np.asarray(
                        [center[0] - (sz - 1) / 2,
                         center[1] - (sz - 1) / 2,
                         center[0] + (sz - 1) / 2 + 1,
                         center[1] + (sz - 1) / 2 + 1], dtype=np.int32)  # shape = (4, )
    
    ## 获得左上角和右下角的坐标
    h, w = img.shape[0], img.shape[1]
    top_left = response_corners[:2]
    bottom_right = response_corners[2:]
    top_left_clamped = np.clip(top_left, 0, [h, w])
    bottom_right_clamped = np.clip(bottom_right, 0, [h, w])
    
    ## 那么, 我们进行裁剪
    dif = get_frame_difference(img[top_left_clamped[0] : bottom_right_clamped[0],
                                    top_left_clamped[1] : bottom_right_clamped[1]],
                               pre_img[top_left_clamped[0] : bottom_right_clamped[0],
                                       top_left_clamped[1] : bottom_right_clamped[1]])
    
    
    
    ## 考虑填充量
    padding = [
        top_left_clamped[0] - top_left[0],  # 上侧填充
        top_left_clamped[1] - top_left[1],  # 左侧填充
        bottom_right[0] - bottom_right_clamped[0],  # 下侧填充
        bottom_right[1] - bottom_right_clamped[1]   # 右侧填充
    ]
    
    ## 应用填充
    if np.any(padding):
        dif = cv2.copyMakeBorder(dif, padding[0], padding[2], padding[1], padding[3],
                                 borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    ## 现在, dif 理当是 (sz, sz) 的图像了
    assert dif.shape[0] == dif.shape[1] == sz
    
    ## 然后，取下 padded_dif 对应位置，与 responses 相乘得到 mixed; 并施加平均值
    movement = cv2.blur(cv2.resize(dif,(cann_len, cann_len), interpolation=cv2.INTER_CUBIC), (19, 19))
    movement = np.abs(movement)
    mixed = response * movement
    
    return movement, mixed

def get_cann_inputs_optimized_2(pre_img, img, responses, szs, center, cann_len):
    '''
    Functions: 返回响应图, 运动图, 混合图, 以及取整后的尺寸(均为 (3, 271, 271))
    '''
    num = szs.shape[0] # 获得图片总数(3张)
    center, szs = np.round(center), num_ops.odd(szs) # 进行取整操作; odd 是保证为奇数
    
    # 找到图中的对应区域，并裁剪下来
    ## 实际上, szs[0] < szs[1] < szs[2]
    ## 首先计算响应图映射回原图像后的边界：(ly, lx, ry, rx)
    response_corners = np.asarray(
                        [center[0] - (szs[-1] - 1) / 2,
                         center[1] - (szs[-1] - 1) / 2,
                         center[0] + (szs[-1] - 1) / 2 + 1,
                         center[1] + (szs[-1] - 1) / 2 + 1], dtype=np.int32)  # shape = (4, )
    
    ## 获得左上角和右下角的坐标
    n, h, w = szs.shape[0], img.shape[0], img.shape[1]
    top_left = response_corners[:2]
    bottom_right = response_corners[2:]
    top_left_clamped = np.clip(top_left, 0, [h, w])
    bottom_right_clamped = np.clip(bottom_right, 0, [h, w])
    
    '''
    首先这里没有必要，请缩放之后再帧差
    '''
    ## 那么, 我们就首先把 szs[2] 的部分在 pre_img, img 中取下来
    dif = get_frame_difference(img[top_left_clamped[0] : bottom_right_clamped[0],
                                    top_left_clamped[1] : bottom_right_clamped[1]],
                               pre_img[top_left_clamped[0] : bottom_right_clamped[0],
                                       top_left_clamped[1] : bottom_right_clamped[1]])
    
    ## 考虑填充量
    padding = [
        top_left_clamped[0] - top_left[0],  # 上侧填充
        top_left_clamped[1] - top_left[1],  # 左侧填充
        bottom_right[0] - bottom_right_clamped[0],  # 下侧填充
        bottom_right[1] - bottom_right_clamped[1]   # 右侧填充
    ]
    
    '''
    这里也没有必要，反正你都填充tmd的0了，你为什么不能缩放之后再填充
    '''
    ## 应用填充
    if np.any(padding):
        dif = cv2.copyMakeBorder(dif, padding[0], padding[2], padding[1], padding[3],
                                 borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    ## 现在, dif 理当是 (szs[2], szs[2]) 的图像了
    assert dif.shape[0] == dif.shape[1] == szs[-1]
    
    ## 所以, 我们按照 szs[0], szs[1] 的大小, 对 dif 进行裁剪
    difs = [dif[(szs[-1] - szs[k]) // 2 : (szs[-1] + szs[k]) // 2,
                (szs[-1] - szs[k]) // 2 : (szs[-1] + szs[k]) // 2] for k in range(num)]
    assert difs[0].shape[0] == difs[0].shape[1] == szs[0]
    assert difs[1].shape[0] == difs[1].shape[1] == szs[1]
    
    ## 然后，取下 padded_dif 对应位置，与 responses 相乘得到 mixed; 并施加平均值
    movements = np.asarray([
        cv2.blur(
            cv2.resize(difs[k],(cann_len, cann_len), interpolation=cv2.INTER_CUBIC), (19, 19)
        )
    for k in range(num)])
    
    movements = np.abs(movements)
    mixeds = responses * movements
    
    return responses, movements, mixeds, szs


def get_cann_inputs_torch(pre_img, img, responses, szs, center, cann_len):
    '''
    Functions: 返回响应图, 运动图, 混合图, 以及取整后的尺寸(均为 (3, 271, 271))
    '''
    num = szs.shape[0] # 获得图片总数(3张)
    center, szs = np.round(center), num_ops.odd(szs) # 进行取整操作; odd 是保证为奇数
    
    # 找到图中的对应区域，并裁剪下来
    ## 实际上, szs[0] < szs[1] < szs[2]
    ## 首先计算响应图映射回原图像后的边界：(ly, lx, ry, rx)
    response_corners = np.asarray(
                        [center[0] - (szs[-1] - 1) / 2,
                         center[1] - (szs[-1] - 1) / 2,
                         center[0] + (szs[-1] - 1) / 2 + 1,
                         center[1] + (szs[-1] - 1) / 2 + 1], dtype=np.int32)  # shape = (4, )
    
    ## 获得左上角和右下角的坐标
    n, h, w = szs.shape[0], img.shape[0], img.shape[1]
    top_left = response_corners[:2]
    bottom_right = response_corners[2:]
    top_left_clamped = np.clip(top_left, 0, [h, w])
    bottom_right_clamped = np.clip(bottom_right, 0, [h, w])
    
    ## 那么, 我们就首先把 szs[2] 的部分在 pre_img, img 中取下来
    dif = get_frame_difference(img[top_left_clamped[0] : bottom_right_clamped[0],
                                    top_left_clamped[1] : bottom_right_clamped[1]],
                               pre_img[top_left_clamped[0] : bottom_right_clamped[0],
                                       top_left_clamped[1] : bottom_right_clamped[1]])
    
    ## 考虑填充量
    padding = [
        top_left_clamped[0] - top_left[0],  # 上侧填充
        top_left_clamped[1] - top_left[1],  # 左侧填充
        bottom_right[0] - bottom_right_clamped[0],  # 下侧填充
        bottom_right[1] - bottom_right_clamped[1]   # 右侧填充
    ]
    
    ## 应用填充
    if np.any(padding):
        dif = cv2.copyMakeBorder(dif, padding[0], padding[2], padding[1], padding[3],
                                 borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    ## 现在, dif 理当是 (szs[2], szs[2]) 的图像了
    assert dif.shape[0] == dif.shape[1] == szs[-1]
    
    ## 所以, 我们按照 szs[0], szs[1] 的大小, 对 dif 进行裁剪
    difs = [dif[(szs[-1] - szs[k]) // 2 : (szs[-1] + szs[k]) // 2,
                (szs[-1] - szs[k]) // 2 : (szs[-1] + szs[k]) // 2] for k in range(num)]
    assert difs[0].shape[0] == difs[0].shape[1] == szs[0]
    assert difs[1].shape[0] == difs[1].shape[1] == szs[1]
    
    ## 然后，取下 padded_dif 对应位置，与 responses 相乘得到 mixed; 并施加平均值
    movements = np.asarray([
        cv2.blur(
            cv2.resize(difs[k],(cann_len, cann_len), interpolation=cv2.INTER_CUBIC), (19, 19)
        )
    for k in range(num)])
    
    mixeds = responses * movements
    
    return responses, movements, mixeds, szs