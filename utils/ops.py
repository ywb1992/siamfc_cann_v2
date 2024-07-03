from __future__ import absolute_import, division

import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from . import num_ops

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')



def init_weights(model, gain=1):
    '''
    Functions: 初始化参数
    '''
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain) # Xavier 初始化方法
            if m.bias is not None:
                nn.init.constant_(m.bias, 0) #常数化初始化
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    '''
    Functions: 读取图片并转化为 RGB 格式
    '''
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img

def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    '''
    Functions: 显示图片, 并在图片上绘制 bounding boxes
    '''
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    
    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale
    
    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]
        
        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])
        
        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)
        
        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)
    
    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img

def show_response(response, pre_coor, gt_coor, colors=[(255, 255, 255), (255, 0, 0), (0, 255, 0)],
                  fig_n=2, delay=1, visualize=True, scale_factor=3):
    '''
    Functions: 显示响应图, 并在图中标记出预测和真实位置
    '''
    response = cv2.resize(response, (response.shape[1] * scale_factor, response.shape[0] * scale_factor))
    ori_response = response
    response = (response - np.min(response)) / (np.max(response) - np.min(response))
    response = (response * 255).astype(np.uint8)
    response = cv2.applyColorMap(response, cv2.COLORMAP_JET)
    
    center_coor = np.array([response.shape[1] // 2,
                            response.shape[0] // 2])
    pre_coor = np.round(pre_coor * scale_factor).astype(np.int32)
    gt_coor = np.round(gt_coor * scale_factor).astype(np.int32)
    
    response = add_arrow(response, center_coor, pre_coor)
    response = add_mark(response, center_coor, colors[0])
    response = add_mark(response, pre_coor, colors[1])
    response = add_mark(response, gt_coor, colors[2])
    
    if visualize:
        
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, response)
        cv2.waitKey(delay)
        mmin = np.min(ori_response) * 1000
        mmax = np.max(ori_response) * 1000
            
        return response # , mmin.item(), mmax.item()
         
def show_response_in_img(img, img_shape, response, response_shape, center, 
                         fig_n=3, delay=1, visualize=True,
                         border_type=cv2.BORDER_CONSTANT,
                         border_value=(0, 0, 0)):
    '''
    Functions: 将响应图映射回原图, 并叠加在原图上
    '''
    # center: (x, y)
    center = np.array([center[1], center[0]])  # (x, y) -> (y, x)
    response_shape = np.round(response_shape).astype(int)
    # 最大最小值归一化，并缩放
    response = (response - np.min(response)) / (np.max(response) - np.min(response))
    response = (response * 255).astype(np.uint8)
    response = cv2.resize(response, tuple(response_shape), interpolation=cv2.INTER_CUBIC)
    response_in_img = np.zeros(img_shape[:2], dtype=np.uint8)
    
    # 这里需要考虑超出边界的情况，根据原文，要进行填充呢
    ## 首先计算响应图映射回原图像后的边界：(ly, lx, ry, rx)
    ### 这里使用 np.floor 是因为 ... center 不一定是整数 ... 
    response_corners = np.asarray(
                        [np.floor(center[0] - (response_shape[0] - 1) / 2),
                        np.floor(center[1] - (response_shape[1] - 1) / 2),
                        np.floor(center[0] + (response_shape[0] - 1) / 2 + 1),
                        np.floor(center[1] + (response_shape[1] - 1) / 2 + 1)])
    ## 然后考虑左上和右下的超出部分有多少
    pads = np.concatenate((- response_corners[0:2], response_corners[2:4] - img_shape[0:2]))
    npad = max(0, int(pads.max())) 
    ## 用三通道均值 RGB = (R_mean, G_mean, B_mean) 将 img 进行填充得到 img'（直到需要裁剪的部分完全在 img' 中）
    if npad > 0:
        response_in_img = cv2.copyMakeBorder(
            response_in_img, npad, npad, npad, npad,
            border_type, value=border_value)
    ## 由于填充，img 的 (0, 0) 已经变成了 img' 中的 (npad, npad)，所以要把矩形框 corners 平移 npad
    ## 获取新的 response 图在 img' 的位置
    response_corners = (response_corners + npad).astype(int)
    ## 填充到 response_in_img 中，注意 response_corners 是 (ly, lx, ry, rx)
    response_in_img[response_corners[0]:response_corners[2], response_corners[1]:response_corners[3]] = response
    response_in_img = response_in_img.astype(np.uint8)
    
    # 再把 response_in_img 裁剪回原图的尺寸
    response_in_img = response_in_img[npad:npad + img_shape[0], npad:npad + img_shape[1]]
    
    # 最后转化回热度图并叠加！结束啦~
    response_in_img = cv2.applyColorMap(response_in_img, cv2.COLORMAP_JET)
    response_in_img = cv2.addWeighted(response_in_img, 0.4, img, 0.6, 0)
    if visualize:
        if response_in_img.shape[0] > 1000 or response_in_img.shape[1] > 1000:
            response_in_img = cv2.resize(response_in_img, None, fx=0.5, fy=0.5)
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, response_in_img)
        cv2.waitKey(delay)
        return response_in_img


def crop_and_resize(img, center, size, out_size,
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
    Functions: 转化为灰度图, 并计算差分
    '''
    img1, img2 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    img1, img2 = img1 / 255.0, img2 / 255.0
    diff = np.abs(img1 - img2)
    return diff

def get_cann_inputs(pre_img, img, responses, szs, center, cann_len):
    '''
    Functions: 返回响应图, 运动图, 混合图, 以及取整后的尺寸(均为 (3, 271, 271))
    '''
    num = szs.shape[0] # 获得图片总数(3张)
    dif = get_frame_difference(pre_img, img) # 获得差分图像
    center, szs = np.round(center), num_ops.odd(szs) # 进行取整操作; odd 是保证为奇数
    responses = responses[:, 0:cann_len, 0:cann_len] # 获取 271 * 271 的响应图像
    
    # 找到图中的对应区域，并裁剪下来
    ## 首先计算响应图映射回原图像后的边界：(ly, lx, ry, rx)
    response_corners = np.asarray(
                        [center[0] - (szs - 1) / 2,
                         center[1] - (szs - 1) / 2,
                         center[0] + (szs - 1) / 2 + 1,
                         center[1] + (szs - 1) / 2 + 1], dtype=np.int32)  # shape = (4, 3)
    response_corners = np.transpose(response_corners) # shape = (3, 4)
    
    ## 然后考虑左上和右下的超出部分有多少
    img_shape = img.shape[0 : 2]
    pads = np.clip(
        np.concatenate([
            -response_corners[:, :2],  # 左上角超出边界的情况
            response_corners[:, 2:] - img_shape  # 右下角超出边界的情况
        ], axis=1), a_min=0, a_max=None
    )
    npad = np.max(pads).astype(np.int32) 
    
    ## 用三通道均值 (0, 0, 0) 将差分图进行填充（直到需要裁剪的部分完全在新图中）
    ## 本来应该这么做的，但是会很慢好吧，所以我打算先裁剪再说
    
    
    padded_dif = cv2.copyMakeBorder(
            dif, npad, npad, npad, npad,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    ## 由于填充，dif 的 (0, 0) 已经变成了 padded 中的 (npad, npad)，所以要把矩形框 corners 平移 npad
    ## 获取新的 response 图在 padded_dif 的位置
    response_corners += npad
    
    ## 此时，把可爱的图图取下来
    
    ## 然后，取下 padded_dif 对应位置，与 responses 相乘得到 mixed; 并施加高斯卷积
    movements = np.asarray([
        cv2.blur(
            cv2.resize(padded_dif[response_corners[k, 0] : response_corners[k, 2],
                                response_corners[k, 1] : response_corners[k, 3]],
                    (cann_len, cann_len), interpolation=cv2.INTER_CUBIC
            ), (19, 19)
        )
    for k in range(num)])
    
    mixeds = responses * movements
    
    return responses, movements, mixeds, szs
       
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

def getMaxPos(x: torch.Tensor):
    return torch.tensor([
        torch.argmax(torch.max(x,1).values,0).item(), torch.argmax(torch.max(x,0).values,0).item()
    ])

def min_max_norm(x: torch.Tensor):
    # x 为一个 3D 的 tensor，一维度是 batch，我们要对每一个矩阵分别 min-max 归一化
    batch_size= x.shape[0]
    min_val = torch.min(x.reshape(batch_size, -1), dim=1, keepdim=True).values.reshape(batch_size, 1, 1)
    max_val = torch.max(x.reshape(batch_size, -1), dim=1, keepdim=True).values.reshape(batch_size, 1, 1)
    # 应用min-max归一化
    x_min_max_normalized = (x - min_val) / (max_val - min_val)
    return x_min_max_normalized

def distribution_norm(x: torch.Tensor):
    # x 为一个 3D 的 tensor，前两个维度是 batch
    # 这里保证要 x 大于 0 才行哟
    batch_size = x.shape[0]
    sum_val = torch.sum(x, dim=[1, 2], keepdim=True)
    # 应用分布归一化
    x_distribution_normalized = x / sum_val
    return x_distribution_normalized

def roll(mat, index, dir, scale_num, len):
    '''
    Functions: 找到了三个尺度中合适的尺度 index 后, 使其进行朝着 dir 方向进行循环位移. 得到的矩阵复制到其它尺度
    Params: index[三个尺度中, 要移动哪一个]; dir[方向]
    '''
    mat = mat[index]
    
    dir = torch.round(dir.squeeze().detach()).to(torch.int32)
    dir = tuple(dir.tolist())
    
    mat = torch.roll(mat, shifts=dir, dims=(0, 1))
    mat = mat.broadcast_to((scale_num, len, len))
    return mat

def add_mark(image, mark_coords, mark_color, 
             x_size=3, circle_radius=6, circle_color=(173, 216, 230), circle_thickness=-1):
    # 指定要标记的坐标
    x, y = mark_coords

    # 画圆形背景
    cv2.circle(image, (x, y), circle_radius, circle_color, circle_thickness)
    
    # 画 X 形状
    thickness = 2

    # 画第一条对角线
    cv2.line(image, (x - x_size, y - x_size), (x + x_size, y + x_size), mark_color, thickness)

    # 画第二条对角线
    cv2.line(image, (x + x_size, y - x_size), (x - x_size, y + x_size), mark_color, thickness)

    return image

def add_arrow(image, start_point, end_point, 
              arrow_color=(0, 0, 255), border_color=(255, 255, 255), border_thickness=1, arrow_thickness=2):
    '''
    在图像上绘制带有边框的箭头。
    
    参数:
    - image: 要绘制箭头的图像
    - start_point: 箭头的起点坐标 (x, y)
    - end_point: 箭头的终点坐标 (x, y)
    - arrow_color: 箭头的颜色 (B, G, R)
    - border_color: 箭头边框的颜色 (B, G, R)
    - border_thickness: 箭头边框的厚度
    - arrow_thickness: 箭头的厚度
    
    返回值:
    - 带有绘制箭头的图像
    '''
    # 先绘制边框
    cv2.arrowedLine(image, start_point, end_point, border_color, arrow_thickness + border_thickness, tipLength=0.3)
    # 再绘制箭头主体
    cv2.arrowedLine(image, start_point, end_point, arrow_color, arrow_thickness, tipLength=0.3)
    return image

       
   