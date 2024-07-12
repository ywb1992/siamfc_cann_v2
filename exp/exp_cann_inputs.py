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

def get_frame_difference_torch(img1, img2):
    '''
    Functions: 接受灰度图, 并计算差分
    '''

    img1_min, img1_max = torch.min(img1), torch.max(img1)
    img2_min, img2_max = torch.min(img2), torch.max(img2)
    
    if img1_min >= 0 and img1_max <= 1 and img2_min >= 0 and img2_max <= 1:
        pass
    else:
        img1, img2 = img1 / 255.0, img2 / 255.0
        
    diff = np.abs(img1 - img2)
    return diff

def ori_get_dif(pre_img, img, sz, center, cann_len):
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
    dif = img_ops.get_frame_difference(img[top_left_clamped[0] : bottom_right_clamped[0],
                                    top_left_clamped[1] : bottom_right_clamped[1]],
                                    pre_img[top_left_clamped[0] : bottom_right_clamped[0],
                                            top_left_clamped[1] : bottom_right_clamped[1]])
    # dif = img[top_left_clamped[0] : bottom_right_clamped[0],
    #           top_left_clamped[1] : bottom_right_clamped[1]]
    
    
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
    
    movement = cv2.blur(cv2.resize(dif,(cann_len, cann_len), interpolation=cv2.INTER_LINEAR), (19, 19))
    
    return movement


def opt_get_dif(pre_img, img, sz, center, cann_len):
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
    dif = img_ops.get_frame_difference(re_img_crop, re_pre_img_crop)
    
    if np.any(re_padding_int):
        dif = cv2.copyMakeBorder(dif, re_padding_int[0], re_padding_int[2], re_padding_int[1], re_padding_int[3],
                                         borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    
    ## 现在, dif 理当是 (sz, sz) 的图像了
    assert dif.shape[0] == dif.shape[1] == cann_len, print(f'({dif.shape[0]}, {dif.shape[1]}), {cann_len}')
    
    movement = cv2.blur(dif, (19, 19))
    
    return movement 

def torch_get_dif(pre_img, img, sz, center, cann_len, device='cuda'):
    '''
        pre_img, img: (1, 1, h, w)
    '''
    center = torch.round(center).to(device)
    sz = num_ops.odd(sz)
    
    response_corners = torch.tensor(
        [
            center[0] - (sz - 1) / 2,
            center[1] - (sz - 1) / 2,
            center[0] + (sz - 1) / 2 + 1,
            center[1] + (sz - 1) / 2 + 1
        ], dtype=torch.int32, device=device
    )

    h, w = img.shape[-2], img.shape[-1]
    top_left = response_corners[:2]
    bottom_right = response_corners[2:]
    top_left_clamped = torch.clamp(top_left, 0, torch.tensor([h, w], device=device))
    bottom_right_clamped = torch.clamp(bottom_right, 0, torch.tensor([h, w], device=device))

    padding = torch.tensor([
        top_left_clamped[0] - top_left[0],
        top_left_clamped[1] - top_left[1],
        bottom_right[0] - bottom_right_clamped[0],
        bottom_right[1] - bottom_right_clamped[1]
    ], device=device)

    re_img_crop = (bottom_right_clamped - top_left_clamped).float() / sz * cann_len
    re_padding = padding.float() / sz * cann_len

    re_img_crop_h = torch.round(re_img_crop[0]).int()
    re_padding_top = torch.round(re_padding[0]).int()
    if re_img_crop_h + re_padding_top > cann_len:
        re_padding_top -= 1
    re_padding_bottom = cann_len - re_img_crop_h - re_padding_top

    re_img_crop_w = torch.round(re_img_crop[1]).int()
    re_padding_left = torch.round(re_padding[1]).int()
    if re_img_crop_w + re_padding_left > cann_len:
        re_padding_left -= 1
    re_padding_right = cann_len - re_img_crop_w - re_padding_left

    re_img_crop_int = (re_img_crop_w, re_img_crop_h)
    re_padding_int = (re_padding_top, re_padding_left, re_padding_bottom, re_padding_right)

    img_crop = img[top_left_clamped[0]:bottom_right_clamped[0], top_left_clamped[1]:bottom_right_clamped[1]].to(device)
    pre_img_crop = pre_img[top_left_clamped[0]:bottom_right_clamped[0], top_left_clamped[1]:bottom_right_clamped[1]].to(device)

    re_img_crop = F.interpolate(img_crop.unsqueeze(0).unsqueeze(0), size=re_img_crop_int, mode='bilinear', align_corners=True).squeeze()
    re_pre_img_crop = F.interpolate(pre_img_crop.unsqueeze(0).unsqueeze(0), size=re_img_crop_int, mode='bilinear', align_corners=True).squeeze()

    dif = get_frame_difference(re_img_crop, re_pre_img_crop)

    if torch.any(torch.tensor(re_padding_int, device=device)):
        dif = F.pad(dif, (re_padding_int[1], re_padding_int[3], re_padding_int[0], re_padding_int[2]), mode='constant', value=0)

    assert dif.shape[0] == dif.shape[1] == cann_len, f'({dif.shape[0]}, {dif.shape[1]}), {cann_len}'

    movement = F.avg_pool2d(dif.unsqueeze(0).unsqueeze(0), kernel_size=19, stride=1, padding=9).squeeze()

    return movement


pre_img = cv2.imread('D:\MyFolders\project\CANN\SiamFC_CANN_v2\data\\train\GOT10K\\train\GOT-10k_Train_000014\\00000001.jpg')
img = cv2.imread('D:\MyFolders\project\CANN\SiamFC_CANN_v2\data\\train\GOT10K\\train\GOT-10k_Train_000014\\00000010.jpg')

pre_img = cv2.cvtColor(pre_img, cv2.COLOR_RGB2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

h, w = img.shape[0], img.shape[1]

steps = 10000

t1, t2, t3 = 0, 0, 0
ori_t, opt_t = 0, 0

np.random.seed(1)

for i in range(steps):
    if i % (steps // 10) == 0:
        print(f'now step: {i}/{steps}')
    center = np.random.rand(2) * np.asarray([h, w])
    sz = np.random.randint(1, np.max([h, w]))
    cann_len = 85
    t1 = time.time()
    ori_movement = ori_get_dif(pre_img, img, sz, center, cann_len)
    t2 = time.time()
    opt_movement = opt_get_dif(pre_img, img, sz, center, cann_len)
    t3 = time.time()
    ori_t += t2 - t1
    opt_t += t3 - t2 
    
    relative_diff = np.abs(ori_movement - opt_movement) / (ori_movement + 1e-15)
    # relative_diff = np.where(relative_diff > 0.2, 1., 0.)
    
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.imshow(ori_movement, cmap='gray')
    ax1.set_title('Original Movement')

    ax2.imshow(opt_movement, cmap='gray')
    ax2.set_title('Optimized Movement')
    
    ax3.imshow(relative_diff, cmap='gray')
    ax3.set_title('Difference')

    plt.tight_layout()
    plt.show()
    
    # assert np.allclose(ori_movement, opt_movement)

print(f'Original Time: {ori_t / steps}, Optimized Time: {opt_t / steps}')


