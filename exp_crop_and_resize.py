import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

project_dir = 'd:/MyFolders/project/CANN/SiamFC_CANN_v2'
sys.path.append(os.path.join(project_dir, 'siamfc_cann_v2/utils'))

from utils import img_ops, num_ops

print(np.inf)

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

img = cv2.imread('D:\MyFolders\project\CANN\SiamFC_CANN_v2\data\\train\GOT10K\\train\GOT-10k_Train_000014\\00000010.jpg')

h, w = img.shape[0], img.shape[1]

steps = 10000

t1, t2, t3 = 0, 0, 0
ori_t, opt_t = 0, 0

np.random.seed(1)

for i in range(steps):
    if i % (steps // 100) == 0:
        print(f'now step: {i}/{steps}')
    center = np.random.rand(2) * np.asarray([h, w])
    size = np.random.randint(1, np.max([h, w]))
    out_size = 255
    avg_color = np.mean(img, axis=(0, 1))
    
    
    t1 = time.time()
    ori_img = crop_and_resize_for_small_image(img, center, size, out_size, border_value=avg_color)
    t2 = time.time()
    opt_img = crop_and_resize(img, center, size, out_size, border_value=avg_color)
    t3 = time.time()
    ori_t += t2 - t1
    opt_t += t3 - t2 

    
    # gray_ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    # gray_opt_img = cv2.cvtColor(opt_img, cv2.COLOR_BGR2GRAY)
    # vague_ori_img = cv2.GaussianBlur(gray_ori_img, (15, 15), 0)
    # vague_opt_img = cv2.GaussianBlur(gray_opt_img, (15, 15), 0)
    # relative_change = np.abs(vague_ori_img.astype(np.int32) - vague_opt_img.astype(np.int32)) / vague_ori_img.astype(np.float32)
    
    # print(f"max relative change: {np.max(relative_change)}")
    # print(f"avg relative change: {np.mean(relative_change)}")
    
    # import matplotlib.pyplot as plt

    # fig, (ax1, ax2) = plt.subplots(2, 1)

    # ax1.imshow(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB))
    # ax1.set_title('Original Movement')

    # ax2.imshow(cv2.cvtColor(opt_img, cv2.COLOR_BGR2RGB))
    # ax2.set_title('Optimized Movement')
    
    # ax3.imshow(relative_change, cmap='gray')
    # ax3.set_title('Relative Change')
    
    # plt.tight_layout()
    # plt.show()
    


print(f'Original Time: {ori_t / steps}, Optimized Time: {opt_t / steps}')


