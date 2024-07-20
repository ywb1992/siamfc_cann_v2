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


def show_image(img, boxes_wh=None, box_fmt='ltwh', colors=None,
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
        if boxes_wh is not None:
            boxes_wh = np.array(boxes_wh, dtype=np.float32) * scale
    
    if boxes_wh is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes_wh = np.array(boxes_wh, dtype=np.int32)
        if boxes_wh.ndim == 1:
            boxes_wh = np.expand_dims(boxes_wh, axis=0)
        if box_fmt == 'ltrb':
            boxes_wh[:, 2:] -= boxes_wh[:, :2]
        
        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes_wh[:, :2] = np.clip(boxes_wh[:, :2], 0, bound)
        boxes_wh[:, 2:] = np.clip(boxes_wh[:, 2:], 0, bound - boxes_wh[:, :2])
        
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
        
        for i, box in enumerate(boxes_wh):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)
    
    if is_visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img

def show_response(response, 
                  fig_n=2, delay=1, scale_factor=3, 
                  peaks_wh=None, peaks_colors=None,
                  is_visualize=True, is_return_value=False):
    '''
    Functions: 显示响应图
    Processing Mode: numpy
    '''
    # 处理响应图
    response = cv2.resize(response, (response.shape[1] * scale_factor, response.shape[0] * scale_factor))
    ori_response = response
    response = (response - np.min(response)) / (np.max(response) - np.min(response))
    response = (response * 255).astype(np.uint8)
    response = cv2.applyColorMap(response, cv2.COLORMAP_JET)
    
    # 获取返回的数值
    mmin = np.min(ori_response)
    mmax = np.max(ori_response)
    
    if is_visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, response)
        cv2.waitKey(delay)

    if is_return_value:
        return response, mmin, mmax
    else:
        return response
def show_response_with_mark(response, pre_coor_wh, gt_coor_wh, colors=[(255, 255, 255), (255, 0, 0), (0, 255, 0)],
                  fig_n=2, delay=1, scale_factor=3, 
                  peaks_wh=None, peaks_colors=None,
                  is_visualize=True, is_return_value=False):
    '''
    Functions: 显示响应图, 并在图中标记出预测和真实位置
    Processing Mode: numpy
    '''
    # 处理响应图
    response = cv2.resize(response, (response.shape[1] * scale_factor, response.shape[0] * scale_factor))
    ori_response = response
    response = (response - np.min(response)) / (np.max(response) - np.min(response))
    response = (response * 255).astype(np.uint8)
    response = cv2.applyColorMap(response, cv2.COLORMAP_JET)
    
    # 获取返回的数值
    mmin = np.min(ori_response)
    mmax = np.max(ori_response)

    # 获得要绘制的三个坐标
    center_coor_wh = np.round(np.array(response.shape[:2]) // 2).astype(np.int32) # 这是因为 response 是 (x, x) 的
    pre_coor_wh = np.round(pre_coor_wh * scale_factor).astype(np.int32)
    gt_coor_wh = np.round(gt_coor_wh * scale_factor).astype(np.int32)
    
    response = add_arrow(response, center_coor_wh, pre_coor_wh)
    response = add_mark(response, center_coor_wh, colors[0])
    response = add_mark(response, pre_coor_wh, colors[1])
    response = add_mark(response, gt_coor_wh, colors[2])
    
    # 人工寻找的最值处
    if peaks_wh is not None:
        if peaks_colors is None:
            peaks_colors = plt.cm.Blues(np.linspace(0.1, 0.9, 5))
            peaks_colors = peaks_colors[::-1]
            peaks_colors = [(int(color[2] * 255), int(color[1] * 255), int(color[0] * 255)) 
                            for color in peaks_colors]
            peaks_colors = peaks_colors[:len(peaks_colors)]
            
        for i, peak_wh in enumerate(peaks_wh):
            peak_wh = np.round(peak_wh * scale_factor).astype(np.int32)
            response = add_mark(response, peak_wh, peaks_colors[i],
                                x_size=1, circle_radius=4, circle_color=(0, 0, 0))
    
    if is_visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, response)
        cv2.waitKey(delay)

    if is_return_value:
        return response, mmin, mmax
    else:
        return response
         
def show_response_in_img(img, img_shape, response, response_shape, center_wh, alpha=0.4,
                         fig_n=3, delay=1,
                         border_type=cv2.BORDER_CONSTANT,
                         border_value=(0, 0, 0),
                         cvt_code=cv2.COLOR_RGB2BGR,
                         is_colored=False, is_visualize=True):
    '''
    Functions: 将响应图映射回原图, 并叠加在原图上; 如有必要, 将会添加上标识
    Processing Mode: numpy
    '''
    # 如果要转化颜色
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    
    # center: hw
    center = np.array([center_wh[1], center_wh[0]])  # (x, y) -> (y, x)
    response_shape = np.round(response_shape).astype(int)
    
    if is_colored is False:
    # 最大最小值归一化
        response = (response - np.min(response)) / (np.max(response) - np.min(response))
        response = (response * 255).astype(np.uint8)
        response = cv2.applyColorMap(response, cv2.COLORMAP_JET)
    
    # 更改尺寸  
    response = cv2.resize(response, tuple(response_shape), interpolation=cv2.INTER_CUBIC)
    
    # 如果不覆盖就叠加
    response_in_img = img
    
    # 这里需要考虑超出边界的情况，根据原文，要进行填充呢
    ## 首先计算响应图映射回原图像后的边界：(ly, lx, ry, rx)
    ### 这里使用 np.floor 是因为 ... center 不一定是整数 ... 
    response_corners = np.asarray([
                        np.floor(center[0] - (response_shape[0] - 1) / 2),
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
    ## 进行透明度的叠加
    img_crop = response_in_img[response_corners[0] : response_corners[2],
                               response_corners[1] : response_corners[3],
                              :]
    response = cv2.addWeighted(response, alpha, img_crop, 1 - alpha, 0)
    response_in_img[response_corners[0] : response_corners[2], 
                    response_corners[1] : response_corners[3],
                   :] = response
    
    # 再把 response_in_img 裁剪回原图的尺寸
    response_in_img = response_in_img[npad : npad + img_shape[0], 
                                      npad : npad + img_shape[1],
                                      :]
    
    if is_visualize:
        if response_in_img.shape[0] > 1000 or response_in_img.shape[1] > 1000:
            response_in_img = cv2.resize(response_in_img, None, fx=0.5, fy=0.5)
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, response_in_img)
        cv2.waitKey(delay)
    return response_in_img

def add_mark(image, mark_coords_wh, mark_color, 
             x_size=3, circle_radius=6, circle_color=(173, 216, 230), circle_thickness=-1,
             mode='hw'):
    '''
    Functions: 在图片上标识一个点
    
    Processing Mode: numpy, (w, h)
    '''
    # 指定要标记的坐标
    x, y = mark_coords_wh

    # 画圆形背景
    cv2.circle(image, (x, y), circle_radius, circle_color, circle_thickness)
    
    # 画 X 形状
    thickness = 2

    # 画第一条对角线
    cv2.line(image, (x - x_size, y - x_size), (x + x_size, y + x_size), mark_color, thickness)

    # 画第二条对角线
    cv2.line(image, (x + x_size, y - x_size), (x - x_size, y + x_size), mark_color, thickness)

    return image

def add_arrow(image, start_point_wh, end_point_wh, 
              arrow_color=(0, 0, 255), border_color=(255, 255, 255), border_thickness=1, arrow_thickness=2):
    '''
    Processing Mode:
        - numpy, (w, h)
        
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
    cv2.arrowedLine(image, start_point_wh, end_point_wh, border_color, arrow_thickness + border_thickness, tipLength=0.3)
    # 再绘制箭头主体
    cv2.arrowedLine(image, start_point_wh, end_point_wh, arrow_color, arrow_thickness, tipLength=0.3)
    return image

def add_minmax_to_image(img, mmin, mmax):
    # 在右上角添加 mmax
    img = add_text_with_background(img, f"mmax: {mmax:.3f}", (img.shape[1] // 2, 25))
    # 在右下角添加 mmin
    img = add_text_with_background(img, f"mmin: {mmin:.3f}",(img.shape[1] // 2, img.shape[0] - 15))
                                            
    return img
       
def add_text_with_background(img, text, position_wh, 
                             font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_color=(0, 255, 255), line_type=2):
    background_color = (0, 0, 0)
    alpha = 0.8
    
    # 获取文字的宽度和高度
    text_size = cv2.getTextSize(text, font, font_scale, line_type)[0]
    
    # 计算背景矩形的坐标
    left_top_wh = (position_wh[0], position_wh[1] - text_size[1] - 10)
    right_bottom_wh = (position_wh[0] + text_size[0] + 10, position_wh[1] + 10)
    
    # 绘制带有透明度的背景矩形
    overlay = img.copy()
    cv2.rectangle(overlay, left_top_wh, right_bottom_wh, background_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # 在背景上添加文字
    cv2.putText(img, text, (position_wh[0], position_wh[1]), font, font_scale, font_color, line_type)
    return img