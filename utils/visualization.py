"""
这个库提供了可视化的所有工具。
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

from . import gen_ops, img_ops, num_ops

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')


def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, is_visualize=True,
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
    
    if is_visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img

def show_response(response, pre_coor, gt_coor, colors=[(255, 255, 255), (255, 0, 0), (0, 255, 0)],
                  fig_n=2, delay=1, is_visualize=True, scale_factor=3):
    '''
    Functions: 显示响应图, 并在图中标记出预测和真实位置
    Processing Mode: numpy
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
    
    if is_visualize:
        
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, response)
        cv2.waitKey(delay)
        mmin = np.min(ori_response) * 1000
        mmax = np.max(ori_response) * 1000
            
        return response # , mmin.item(), mmax.item()
         
def show_response_in_img(img, img_shape, response, response_shape, center, 
                         fig_n=3, delay=1, is_visualize=True,
                         border_type=cv2.BORDER_CONSTANT,
                         border_value=(0, 0, 0)):
    '''
    Functions: 将响应图映射回原图, 并叠加在原图上
    Processing Mode: numpy
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
    if is_visualize:
        if response_in_img.shape[0] > 1000 or response_in_img.shape[1] > 1000:
            response_in_img = cv2.resize(response_in_img, None, fx=0.5, fy=0.5)
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, response_in_img)
        cv2.waitKey(delay)
        return response_in_img

def add_mark(image, mark_coords, mark_color, 
             x_size=3, circle_radius=6, circle_color=(173, 216, 230), circle_thickness=-1):
    '''
    Functions: 在图片上标识一个点
    
    Processing Mode: numpy
    '''
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
    Processing Mode:
        - numpy
        
    Functions:
        - 在图像上绘制带有边框的箭头。
    
    Parameters:
        - image: 要绘制箭头的图像
        - start_point: 箭头的起点坐标 (x, y)
        - end_point: 箭头的终点坐标 (x, y)
        - arrow_color: 箭头的颜色 (B, G, R)
        - border_color: 箭头边框的颜色 (B, G, R)
        - border_thickness: 箭头边框的厚度
        - arrow_thickness: 箭头的厚度
    
    Returns:
        - 带有绘制箭头的图像
    
    '''
    # 先绘制边框
    cv2.arrowedLine(image, start_point, end_point, border_color, arrow_thickness + border_thickness, tipLength=0.3)
    # 再绘制箭头主体
    cv2.arrowedLine(image, start_point, end_point, arrow_color, arrow_thickness, tipLength=0.3)
    return image

       
   