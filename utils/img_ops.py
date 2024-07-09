"""
这个库提供了针对图片和标注框的操作，包括读取、裁剪、帧差等。
"""

from __future__ import absolute_import, division

import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from . import gen_ops, num_ops

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')


def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    '''
    Functions: 读取图片并转化为 RGB 格式
    '''
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img

def crop_and_resize_for_small_image(img, center, size, out_size,
                              border_type=cv2.BORDER_CONSTANT,
                              border_value=(0, 0, 0),
                              interp=cv2.INTER_LINEAR):
    '''
    Functions: 优化的图像裁剪并缩放功能。
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

def crop_and_resize(img, center, sz, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR):
    '''
    Functions: 优化的图像裁剪并缩放功能。
    先缩放，再裁剪，如果裁剪区域部分在图像外，则对裁剪后的图像进行填充。
    '''
    if sz < out_size:
        return crop_and_resize_for_small_image(img, center, sz, out_size, border_type, border_value, interp)
    
    center, sz = np.round(center), num_ops.odd(sz)
    ## 首先计算响应图映射回原图像后的边界：(ly, lx, ry, rx)
    corners = np.asarray(
                        [center[0] - (sz - 1) / 2,
                         center[1] - (sz - 1) / 2,
                         center[0] + (sz - 1) / 2 + 1,
                         center[1] + (sz - 1) / 2 + 1], dtype=np.int32)  # shape = (4, )
    
    ## 获得左上角和右下角的坐标
    h, w = img.shape[0], img.shape[1]
    top_left = corners[:2]
    bottom_right = corners[2:]
    top_left_clamped = np.clip(top_left, 0, [h, w])
    bottom_right_clamped = np.clip(bottom_right, 0, [h, w])
    
    ## 考虑填充量
    padding = np.asarray([
        top_left_clamped[0] - top_left[0],  # 上侧填充
        top_left_clamped[1] - top_left[1],  # 左侧填充
        bottom_right[0] - bottom_right_clamped[0],  # 下侧填充
        bottom_right[1] - bottom_right_clamped[1]   # 右侧填充
    ])
    
    ## 考虑需要将裁剪下来的图片缩放到什么尺寸
    re_img_crop = 1. * (bottom_right_clamped - top_left_clamped) / sz * out_size
    re_padding = 1. * padding / sz * out_size
    
    ### 对 re_img_crop 和 re_padding 进行取整操作，但保证取整之后和不变
    re_img_crop_h = np.round(re_img_crop[0]).astype(np.int32)
    re_padding_top = np.round(re_padding[0]).astype(np.int32)
    if re_img_crop_h + re_padding_top > out_size:
        re_padding_top -= 1
    re_padding_bottom = out_size - re_img_crop_h - re_padding_top
    
    re_img_crop_w = np.round(re_img_crop[1]).astype(np.int32)
    re_padding_left = np.round(re_padding[1]).astype(np.int32)
    if re_img_crop_w + re_padding_left > out_size:
        re_padding_left -= 1
    re_padding_right = out_size - re_img_crop_w - re_padding_left
    
    re_img_crop_int = (re_img_crop_w, re_img_crop_h) # 注意，resize 接受的是 (w, h), ***
    re_padding_int = (re_padding_top, re_padding_left, re_padding_bottom, re_padding_right)
    
    
    ## 那么, 我们进行裁剪，而后放缩，再考虑填充
    img_crop = img[top_left_clamped[0] : bottom_right_clamped[0],
                   top_left_clamped[1] : bottom_right_clamped[1], :]
    re_img_crop = cv2.resize(img_crop, re_img_crop_int, 
                             interpolation=interp)

    if np.any(re_padding_int):
        re_img_crop = cv2.copyMakeBorder(re_img_crop, re_padding_int[0], re_padding_int[2], re_padding_int[1], re_padding_int[3],
                                         border_type, value=border_value)
    
    return re_img_crop

def img_to_tensor(img):
    '''
    Functions: 接受 img, 并转化为 (1, 3, h, w) 形式的张量
    '''
    if img.shape[2] == 3:
        img = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
    elif img.shape[3] == 3:
        img = torch.tensor(img).permute(0, 3, 1, 2).float()
    else:
        raise ValueError('The shape of img is not correct!', img.shape)
    return img.to(device)

def get_center_sz(box):
    '''
    Functions: 获取 ltwh 标注格式的 box 或者 boxes 的中心位置
    '''
    if box.ndim == 1:
        box = np.array([
                box[0] - 1 + (box[2] - 1) / 2,
                box[1] - 1 + (box[3] - 1) / 2,
                box[2], box[3]], dtype=np.float32)
        return box[:2], box[2:]
    elif box.ndim == 2:
        box = np.array([
                box[:, 0] - 1 + (box[:, 2] - 1) / 2,
                box[:, 1] - 1 + (box[:, 3] - 1) / 2,
                box[:, 2], box[:, 3]], dtype=np.float32)
        box = box.transpose()
        return box[:, :2], box[:, 2:]

def get_instance(img, instance_sz, 
                 predicted_center, crop_sz,
                 scale_factors):
    '''
    Functions: 从图像中裁剪出 instance, 缩放为 3 * 255 * 255, 并返回其张量形式
    '''
    avg_color = np.mean(img, axis=(0, 1))
    instance = [crop_and_resize(
        img, predicted_center, crop_sz * f,
        instance_sz, border_value=avg_color 
    ) for f in scale_factors]
    instance = np.stack(instance, axis=0)
    instance = img_to_tensor(instance)
    return instance

def get_frame_difference(img1, img2):
    '''
    Functions: 接受灰度图, 并计算差分
    '''

    img1_min, img1_max = np.min(img1), np.max(img1)
    img2_min, img2_max = np.min(img2), np.max(img2)
    
    if img1_min >= 0 and img1_max <= 1 and img2_min >= 0 and img2_max <= 1:
        pass
    else:
        img1, img2 = img1 / 255.0, img2 / 255.0
        
    diff = np.abs(img1 - img2)
    return diff

def get_cann_inputs(pre_img, img, response, sz, center, cann_len):
    '''
    Functions: 返回运动图, 混合图, 以及取整后的尺寸(均为 (271, 271))
    '''
    center, sz = np.round(center), num_ops.odd(sz) # 进行取整操作; odd 是保证为奇数
    
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
    
    ## 考虑填充量
    padding = np.asarray([
        top_left_clamped[0] - top_left[0],  # 上侧填充
        top_left_clamped[1] - top_left[1],  # 左侧填充
        bottom_right[0] - bottom_right_clamped[0],  # 下侧填充
        bottom_right[1] - bottom_right_clamped[1]   # 右侧填充
    ])
    
    ## 考虑需要将裁剪下来的图片缩放到什么尺寸
    re_img_crop = 1. * (bottom_right_clamped - top_left_clamped) / sz * cann_len
    re_padding = 1. * padding / sz * cann_len
    
    ### 对 re_img_crop 和 re_padding 进行取整操作，但保证取整之后和不变
    re_img_crop_h = np.round(re_img_crop[0]).astype(np.int32)
    re_padding_top = np.round(re_padding[0]).astype(np.int32)
    if re_img_crop_h + re_padding_top > cann_len:
        re_padding_top -= 1
    re_padding_bottom = cann_len - re_img_crop_h - re_padding_top
    
    re_img_crop_w = np.round(re_img_crop[1]).astype(np.int32)
    re_padding_left = np.round(re_padding[1]).astype(np.int32)
    if re_img_crop_w + re_padding_left > cann_len:
        re_padding_left -= 1
    re_padding_right = cann_len - re_img_crop_w - re_padding_left
    
    re_img_crop_int = (re_img_crop_w, re_img_crop_h) # 注意，resize 接受的是 (w, h), ***
    re_padding_int = (re_padding_top, re_padding_left, re_padding_bottom, re_padding_right)
    
    
    ## 那么, 我们进行裁剪，而后放缩，再考虑填充
    img_crop = img[top_left_clamped[0] : bottom_right_clamped[0],
                   top_left_clamped[1] : bottom_right_clamped[1]]
    pre_img_crop = pre_img[top_left_clamped[0] : bottom_right_clamped[0],
                           top_left_clamped[1] : bottom_right_clamped[1]]
    re_img_crop = cv2.resize(img_crop, re_img_crop_int, 
                             interpolation=cv2.INTER_LINEAR)
    re_pre_img_crop = cv2.resize(pre_img_crop, re_img_crop_int,
                                 interpolation=cv2.INTER_LINEAR)
    dif = get_frame_difference(re_img_crop, re_pre_img_crop)
    
    if np.any(re_padding_int):
        dif = cv2.copyMakeBorder(dif, re_padding_int[0], re_padding_int[2], re_padding_int[1], re_padding_int[3],
                                         borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    
    ## 现在, dif 理当是 (sz, sz) 的图像了
    assert dif.shape[0] == dif.shape[1] == cann_len, print(f'({dif.shape[0]}, {dif.shape[1]}), {cann_len}')
    
    movement = cv2.blur(dif, (19, 19))
    mixed = response * movement
    
    return movement, mixed  

def upsample(img_tensor, from_scale, to_scale):
    '''
    Functions: 执行上采样, 或者说图片放大, 从 from_scale 到 to_scale
    '''
    # 假设 imgs 是一个形状为 (bs, 1, 17, 17) 的四维张量
    img_numpy = img_tensor.cpu().numpy()
    batch_size = img_tensor.shape[0]
    channels = img_tensor.shape[1]
    # 创建一个空的数组来存放上采样后的图像
    upsampled_imgs = np.zeros((batch_size, channels, to_scale, to_scale), dtype=np.float32)

    # 遍历批量中的每个图像，并分别进行上采样
    for batch in range(batch_size):
        for channel in range(channels):
            # 从 (17, 17) 上采样到 (272, 272)
            upsampled_imgs[batch, channel] = cv2.resize(
                img_numpy[batch, channel], (to_scale, to_scale), interpolation=cv2.INTER_CUBIC)
    
    upsampled_imgs = torch.tensor(upsampled_imgs, device=device)
    return upsampled_imgs




###########################
##### 已经不使用的函数 #####
###########################

def get_cann_inputs_old(pre_img, img, response, sz, center, cann_len):
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
    mixed = response * movement
    
    return movement, mixed  
