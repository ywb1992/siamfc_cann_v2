from __future__ import absolute_import, division, print_function

import json
import os
import sys
import time
from collections import namedtuple
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from got10k.trackers import Tracker
from siamfc import TrackerSiamFC
from utils import calc, ops, ops_torch
from utils.video_save import *

from .cann_dataloader import my_collate_fn
from .cann_datasets import CANN_Pair
from .cann_losses import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
st1, _t1 = 0, 0
st2, _t2 = 0, 0
st3, _t3 = 0, 0

class CANN_Network(nn.Module):
    '''
    Functions: 定义 CANN
    '''
    def __init__(self, len=271, tau=2.0, A=1.2, k=0.1, a=0.6, 
                 factor0=0.3, factor1=2.0, factor2=2.0, mix_factor=0.95,
                 scale_num=3):
        '''
        Functions: 设定初始参数. 
        Params: len[CANN大小]; tau[时间常数]; A[连接常数]; k[抑制常数];
                a[空间常数]; factor0[mu 系数]; factor1[mot 系数]; factor2[mix 系数]; mix_factor[混合 U 与 res 的系数];
                scale_num[多尺度的数量, 这里为 3]
        '''
        super(CANN_Network, self).__init__()
        
        # 下面是基础参数设置
        self.scale_num = scale_num
        self.len = torch.tensor(len, device=device)
        self.shape = torch.tensor([self.len, self.len], device=device)
        self.N = torch.tensor(len * len, device=device)
        self.range = torch.tensor(2 * torch.pi, device=device)
        self.rho = self.len * self.len / (self.range * self.range)
        self.x = torch.linspace(0, self.range, self.len, device=device)
        self.y = torch.linspace(0, self.range, self.len, device=device)
        self.u = torch.zeros((self.scale_num, self.len, self.len), device=device)
        self.r = torch.zeros((self.scale_num, self.len, self.len), device=device)
        
        # 需要保证这些参数在反向传播时非负，所以先设定为平方根，update_para 时再平方
        self.sqrt_tau = nn.Parameter(torch.sqrt(torch.tensor(tau, device=device)))
        self.sqrt_A = nn.Parameter(torch.sqrt(torch.tensor(A, device=device)))
        self.sqrt_k = nn.Parameter(torch.sqrt(torch.tensor(k, device=device)))
        # self.a = torch.tensor(a, device=device)
        self.sqrt_a = nn.Parameter(torch.sqrt(torch.tensor(a, device=device)))
        self.sqrt_factor0 = nn.Parameter(torch.sqrt(torch.tensor(factor0, device=device)))
        self.sqrt_factor1 = nn.Parameter(torch.sqrt(torch.tensor(factor1, device=device)))
        self.sqrt_factor2 = nn.Parameter(torch.sqrt(torch.tensor(factor2, device=device)))
        self.sqrt_mix_factor = nn.Parameter(torch.sqrt(torch.tensor(mix_factor, device=device)))
        
        # 获取参数
        self.update_para()
        
    def update_para(self):
        """
        Functions: 获取存在梯度的参数
        """
        # 平方根变为平方
        self.tau = torch.square(self.sqrt_tau)
        self.A = torch.square(self.sqrt_A)
        self.k = torch.square(self.sqrt_k)
        self.a = torch.square(self.sqrt_a)
        self.factor0 = torch.square(self.sqrt_factor0)
        self.factor1 = torch.square(self.sqrt_factor1)
        self.factor2 = torch.square(self.sqrt_factor2)
        self.mix_factor = torch.square(self.sqrt_mix_factor)
        # self.mix_factor = nn.Parameter(torch.sqrt(torch.tensor(0.90, device=device)))
        
        # CANN稳定时，u 和 r 的峰值为 u_0 和 r_0
        self.k_c = self.A ** 2 * self.rho / (32.0 * torch.pi * self.a ** 2)
        self.u_0 = (1 + torch.sqrt(1 - self.k / self.k_c)) * self.A / (8 * torch.pi * self.a ** 2 * self.k)
        self.r_0 = (1 + torch.sqrt(1 - self.k / self.k_c)) / (4 * torch.pi * self.a ** 2 * self.k * self.rho)
        
        def create_conv_kernel(len):
            '''
            Functions: 创建一个高斯卷积核, shape=(3, len, len)
            '''
            center = torch.tensor([len // 2, len // 2], device=device)
            center = center.broadcast_to((self.scale_num, 2))
            kernel = self.create_dis_mat(center)
            kernel = self.A * (torch.exp(-0.5 * torch.square(kernel / self.a)) / 
                            (2 * torch.tensor(torch.pi) * self.a ** 2))       
            return kernel
        
        self.kernel = create_conv_kernel(self.len) # 创建连接矩阵 J 的卷积核形式
        self.fft_kernel = torch.fft.fft2(self.kernel) # 进行傅里叶变换
        
    def forward(self, input, dt):
        '''
        Functions: 输入 input, 进行 dt 时间步长的动力学离散模拟
        Params: input[输入电流]; dt[时间步长]
        '''

        fft_r = torch.fft.fft2(self.r) # f(r)
        fft_conn_input = fft_r * self.fft_kernel # f(J) * f(r)
        conn_input = (torch.fft.ifft2(fft_conn_input)).real # J \conv r  
        conn_input = torch.roll(conn_input, shifts=(self.len // 2 + 1, self.len // 2 + 1), dims=(1, 2)) # 处理卷积后位置不匹配
        self.u = self.u + dt * (-self.u + conn_input + input) / self.tau # 更新 u, 一个步长的模拟
        self.r = self.u ** 2 / (1.0 + self.k * torch.sum(self.u ** 2, dim=(1, 2), keepdim=True)) # 更新 r
        # print((self.k * torch.sum(self.u ** 2, dim=(1, 2), keepdim=True)[0]).item())
        
        return self.u
        
    def create_dis_mat(self, center):
        '''
            Functions: 以 center 为中心创建一个距离矩阵, 距离的定义为圆环上的距离, shape=(3, len, len)
        '''
        if center.device != device:
            center = center.to(device)
        center = center / self.len * self.range
        
        x_coords, y_coords = torch.meshgrid(torch.arange(self.len), torch.arange(self.len), indexing='ij')
        x_coords, y_coords = x_coords.to(device), y_coords.to(device)
        x_coords, y_coords = x_coords / self.len * self.range, y_coords / self.len * self.range # 缩放到 [0, 2pi)
        x_coords, y_coords = x_coords.broadcast_to((self.scale_num, self.len, self.len)), \
                             y_coords.broadcast_to((self.scale_num, self.len, self.len))
        x_coords, y_coords = torch.abs(x_coords - (center[:, 0]).unsqueeze(-1).unsqueeze(-1)), \
                             torch.abs(y_coords - (center[:, 1]).unsqueeze(-1).unsqueeze(-1)) # 考虑与中心的欧式距离
        x_coords, y_coords = torch.where(x_coords > self.range / 2, self.range - x_coords, x_coords), \
                             torch.where(y_coords > self.range / 2, self.range - y_coords, y_coords) # 考虑圆环距离
        dis_mat = torch.sqrt((x_coords)**2 + (y_coords)**2) # 综合 x 和 y 的距离信息
        return dis_mat
    
    def set_stable(self, center):
        '''
        Functions: 设置一个已经稳定的 CANN
        Params: center[高斯峰中心]
        '''
        if center.device != device:
            center = center.to(device)
        dis_mat = self.create_dis_mat(center)
        self.u = self.u_0 * torch.exp(-dis_mat ** 2 / (4 * self.a ** 2)) # 设置 u, 吴思老师的公式
        self.r = self.u ** 2 / (1.0 + self.k * torch.sum(self.u ** 2, dim=(1, 2), keepdim=True)) # 设置 r
        pass
    
    def roll(self, index, dir):
        '''
        Functions: 找到了三个尺度中合适的尺度 index 后, 使其进行朝着 dir 方向进行循环位移. 得到的 u 复制到其它尺度
        Params: index[三个尺度中, 要移动哪一个]; dir[方向]
        '''
        cann_u, cann_r = self.u[index], self.r[index]
        
        dir = torch.round(dir.squeeze().detach()).to(torch.int32)
        dir = tuple(dir.tolist())
        
        cann_u = torch.roll(cann_u, shifts=dir, dims=(0, 1))
        cann_r = torch.roll(cann_r, shifts=dir, dims=(0, 1))
        
        self.u = cann_u.broadcast_to((self.scale_num, self.len, self.len))
        self.r = cann_r.broadcast_to((self.scale_num, self.len, self.len))
    
class CANN_Runner:
    '''
    Functions: 用于给予 CANN 以输入
    '''
    def __init__(self, CANN: CANN_Network):
        self.CANN = CANN
    def set_input_directly(self, input):
        self.input = input.to(device)
    def execute(self, dt):
        self.CANN(self.input, dt)

class CANN_Tracker(Tracker):
    '''
    Functions: SiamFC+CANN 追踪器，包含了追踪(评估)模块与训练模块
    '''
    def __init__(self, net_path=None, cann_path=None, failure_path=None, mode='test', tracker_name='SiamFC_CANN', **kwargs):
        '''
        Functions: 初始化所有基本参数; 父类为 got10k.trackers.Tracker. 父类的方法里初始化了所有超参数: self.cfg
        Params: net_path[siamfc 的预训练模型路径]; cann_path[CANN 的预训练模型路径]; failure_path[失败视频的保存路径]
        '''
        # 初始化父类，这里定义了所有的超参数: self.cfg
        super(CANN_Tracker, self).__init__(tracker_name, True, **kwargs) 
        
        # 这里根据任务调整 batch_size
        assert mode in ['test', 'eval', 'train']
        if mode == 'test' or mode == 'eval':
            self.cfg = self.cfg._replace(batch_size=1)
        
        # 设置保存路径的信息
        now = datetime.now()
        self.formatted_date = now.strftime("%Y_%m_%d_%H_%M_%S") # 获取当前时间，用于保存模型

        # 加载 cann 模型, 若有预训练则加载
        self.net = CANN_Network(self.cfg.len, self.cfg.tau, self.cfg.A,
                                self.cfg.k, self.cfg.a, 
                                self.cfg.factor0, self.cfg.factor1, self.cfg.factor2, self.cfg.mix_factor,
                                self.cfg.scale_num)
        if cann_path is not None:
            self.net.load_state_dict(torch.load(
                cann_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(device)
        self.net.update_para()
        
        # 加载 cann 的 runner 和 siamfc 模型
        self.runner = CANN_Runner(self.net)
        self.siamfc = TrackerSiamFC(net_path=net_path)
        
        # 冻结 siamfc 参数, 解冻 cann 参数
        for param in self.siamfc.net.parameters():
            param.requires_grad_(False)
        for param in self.net.parameters():
            param.requires_grad_(True)
        
        # 设置损失函数
        self.criterion = CenterDistanceLoss()

        # 设置 Adam 优化器
        self.optimizer = optim.Adam(self.net.parameters(), 
                                    lr=self.cfg.initial_lr,
                                    weight_decay=self.cfg.weight_decay)
        '''
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay, # 权重衰减：防止过拟合
            momentum=self.cfg.momentum) # 动量参数：加速收敛
        '''    
        
        # 设置指数下降学习器
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num) # (终止/起始) ^ (1/epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma) # 指数衰减的学习率, lr = lr_0 * (gamma^epoch)
        
        # 设置训练和推理都需要的参数
        # 17 * 17 的响应图会被上采样 16 倍, 扩大为 272 * 272, 以方便寻找最大值
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz 
        # 缩放因子, 搜寻区域的多尺度搜索(这里为 3 个尺度)
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num) # 0.964, 1.000, 1.0375 
        # 惩罚因子, 针对放大和缩小了的尺度
        self.scale_penalty = [self.cfg.scale_penalty, 1.0, self.cfg.scale_penalty]
        
        # 汉明余弦窗, 用于中心抑制
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum() # 分布归一化
    
    def init(self, img, box, is_train=False):
        '''
        Functions: 根据初始帧 img 和标注框 box 初始化追踪器, 并设置为训练或评估模式
        Params: img[第一帧图像]; box[第一帧的标注框]; is_train[是否为训练模式]
        '''
        
        # cann 设置成训练或者评估模式
        if is_train == False:
            self.net.eval()
            for param in self.net.parameters():
                param.requires_grad_(False)
        else:
            self.net.train()
            for param in self.net.parameters():
                param.requires_grad_(True)
                
        # 再次设置, 保证 siamfc 无需训练
        self.siamfc.net.eval()
        self.siamfc.net.to(device)
        for param in self.siamfc.net.parameters():
            param.requires_grad_(False)
        
        # 271 * 271 矩阵的中心, 值为 (1, 135, 135)
        self.constant_center = torch.tensor([[self.cfg.len // 2, self.cfg.len // 2]])
        self.net.update_para()
        self.net.set_stable(self.constant_center.broadcast_to(
            (self.cfg.scale_num, 2)).to(device)) # 初始化稳定在矩阵中心的 cann
        
        # box 在这里有两种格式, 1. (给定的格式/输入和应当输出的格式) 和 2. 内部处理的格式
        # 给定 box 的格式为 ltwh, 也即 (x, y, w, h)/(左上角, 高, 宽)
        # 处理 box 的格式是 chw,  (cy, cx, h, w), 也就是(中心点, 宽, 高)
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:] # 获得第一帧的中心点 (cy, cx) 以及尺寸 (h, w)
        self.center = torch.tensor(self.center)
        
        # 获得裁剪全图时: 初始帧裁剪区域 z 的大小, 搜索帧裁剪区域 x 的大小 
        context = self.cfg.context * np.sum(self.target_sz) # context = 0.5 * (h + w)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context)) # z_sz = sqrt((h + context) * (w + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz # 按比例缩放，系数即为 255 / 127
        
        
        # 然后根据上面的尺寸，裁剪出 z_sz 大小的图像，然后填充并缩放成 127 * 127
        self.avg_color = np.mean(img, axis=(0, 1)) # 获得三通道的均值，用于裁剪区域超出图像时的填充
        z = ops.crop_and_resize(
            img, self.center.detach().numpy(), self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)
        
        # 样本图进行一些维度处理, (h, w, 3) -> (1, 3, h, w)
        z = ops.img_to_tensor(z)
        
        # 根据样本图获得卷积核 kernel, size = (1, 256, 6, 6) 
        self.kernel = self.siamfc.net.backbone(z)
        
        # 设置上一个输入
        self.last_input = None
    
    def track(self, img_files, box, annos, seq_name, 
              is_train=False, visualize=False, is_record_delta=False):
        '''
        Functions: 进行一整个序列的追踪, 进行评估或者训练。
        Params: img_files[图像序列的路径]; box[第一帧的标注框]; anno[所有帧的标注框]; seq_name[视频序列的名字];
                is_train[是否训练]; 
                visualize[是否可视化]; 
                is_record_delta[是否记录中心误差]
        '''
        # 初始化一些参数
        frame_num = len(img_files) # 有多少帧
        boxes = np.zeros((frame_num, 4)) # 此为输出标注框, 格式为 ltwh/(lx, ly, w, h)
        boxes[0] = box # 初始化第一帧的标注框
        times = np.zeros(frame_num) # 用于记录运行时间
        pre_img = None # 用于保存上一帧图像
        imgs = [] # 用于保存每一帧的可视化图像(可视化了标注框)
        
        if visualize: # 初始化画布
            # self.fig, self.ax = plt.subplots(2, 3, figsize=(10, 15))
            pass
        if is_record_delta:
            delta = []
        if is_train:
            total_turns = len(img_files) # 总追踪轮数
            video_ce2 = 0 # 整个视频的中心误差平方总和
            video_ce = 0 # 整个视频的中心误差总和
            ce2_per_frame_cann = [] # 每帧的中心误差平方(CE^2)
            ce_per_frame_cann = [] # 每帧的中心误差(CE)
            stop_recording = False
        
        # 遍历每一帧图像, 并追踪
        for f, img_file in enumerate(img_files): 
            img = ops.read_image(img_file) # 读入图像，格式为: RGB, [0, 255], shape=(h, w, 3)
            begin = time.time()
        
            if f == 0: # 第一帧，抽取模板图像，生成卷积核
                self.init(img, box, is_train=is_train) 
                sz = 1
            else: # 后续帧：生成响应图，生成标注框
                if pre_img.shape != img.shape: # 有些训练数据里面有问题，需要跳过
                    break 
                boxes[f], sz = self.update(pre_img, img, 
                                            is_train=is_train, visualize=visualize) # 更新标注框
                
            pre_img = img
            times[f] = time.time() - begin
            
            if is_record_delta: # 如果需要记录中心误差，那么每次记录 (del_in_res, del_in_img)
                search_center, _ = ops.get_center_sz(boxes[f - 1])
                gt_center, _ = ops.get_center_sz(annos[f])
                gt_disp_in_img = search_center - gt_center
                gt_disp_in_res = gt_disp_in_img * self.cfg.len / sz
                delta.append(
                    (np.linalg.norm(gt_disp_in_res), np.linalg.norm(gt_disp_in_img))
                )    
            
            if is_train: # 如果处于训练模式，那么记录中心误差并反向传播
                gt_center, _ = ops.get_center_sz(annos[f])
                frame_ce2 = self.criterion(self.center, torch.from_numpy(gt_center)) # 获取含有梯度的中心误差, ce ** 2
                frame_ce = torch.sqrt(frame_ce2).item() # 获取中心误差, ce
                
                if stop_recording == False:
                    video_ce2 += frame_ce2
                    video_ce += frame_ce
                else:
                    video_ce2 += frame_ce2.detach()
                    video_ce += frame_ce
                
                ce2_per_frame_cann.append(frame_ce2.item()) # 记录每帧的中心误差平方
                ce_per_frame_cann.append(frame_ce) # 记录每帧的中心误差
                
                # 格式转化, 从 ltwh/(lx, ly, w, h) 转化为 ltrb/(lx, ly, rx, ry)
                p_box = np.array([
                    boxes[f][0], boxes[f][1],
                    boxes[f][0] + (boxes[f][2] - 1), boxes[f][1] + (boxes[f][3] - 1)
                ]) 
                a_box = np.array([
                    annos[f][0], annos[f][1],
                    annos[f][0] + (annos[f][2] - 1), annos[f][1] + (annos[f][3] - 1)
                ])
                
                IoU = box_iou(torch.from_numpy(p_box).unsqueeze(0),
                              torch.from_numpy(a_box).unsqueeze(0))
                if IoU.item() < self.cfg.IoU_thresold:
                    stop_recording = True
            
            if visualize:
                img = ops.show_image(img, [boxes[f, :], annos[f, :]]) 
        
            
                
        # 追踪完整个序列后, 和 siamfc 的追踪效果进行比较, 并进行反向传播(但尚不更新梯度)
        if is_train:
            siamfc_boxes, _, _ = self.siamfc.track(img_files, box, annos, "test", False, False)
            
            siamfc_boxes_center, _ = ops.get_center_sz(siamfc_boxes)
            gt_center, _ = ops.get_center_sz(annos)
            
            ce_per_frame_siamfc = [np.linalg.norm(siamfc_boxes_center[i] - gt_center[i]) 
                                   for i in range(len(siamfc_boxes_center))]
            ce2_per_frame_siamfc = [ce ** 2 for ce in ce_per_frame_siamfc]

            siamfc_avg_ce2 = np.sum(ce2_per_frame_siamfc) / total_turns
            cann_avg_ce2 = video_ce2 / total_turns
            
            siamfc_avg_ce = np.sqrt(siamfc_avg_ce2)
            cann_avg_ce = video_ce / total_turns
            
            print("tunrs: {}/{}, cann_ce: {}, siamfc_ce: {}".format(
                total_turns, len(img_files), cann_avg_ce, siamfc_avg_ce
            ))
            return cann_avg_ce2, siamfc_avg_ce2, cann_avg_ce, siamfc_avg_ce
        else:     
            if is_record_delta:
                return boxes, times, imgs, delta
            else:       
                return boxes, times, imgs, None # 返回[每一帧的标注框], [每一帧图像的用时], [每一帧的可视化图像](可能为空)
    
    def update(self, pre_img, img, is_train=False, visualize=False, video_saving=False): 
        '''
        Functions: 进行当前帧标注框的预测
        '''
        # 第一步：获得 responses
        instance = ops.get_instance(
            img, self.cfg.instance_sz, self.center.detach().numpy(), 
            self.x_sz, self.scale_factors
        )
        responses = self.get_resized_response(self.kernel, instance) # 获取 (3, 1, 272, 272) 的响应图
        responses = responses.squeeze() # 压缩维度, (3, 272, 272)
        responses = responses * torch.tensor(self.scale_penalty, device=device).unsqueeze(-1).unsqueeze(-1) # 施加尺度的惩罚因子

        # 第二步：获得 cann 的 inputs
        szs_in_img = self.upscale_sz * (self.cfg.total_stride / self.cfg.response_up) \
                                        * (self.x_sz * self.scale_factors) / self.cfg.instance_sz # 获得响应图在原图中的对应尺寸
        
        _responses, movements, mixed, szs = ops_torch.get_cann_inputs_optimized(
            pre_img, img,
            responses.cpu().detach().numpy(), szs_in_img,
            self.center.detach().numpy(), self.cfg.len
        )
        
        
        # 第三步：送入 CANN 进行处理
        movements_tensor = torch.from_numpy(movements).to(device)
        responses_tensor = torch.from_numpy(_responses).to(device)
        mixed_tensor = torch.from_numpy(mixed).to(device)
        
        inputs_tensor = 0.05 * responses_tensor + self.net.factor1 * movements_tensor + self.net.factor2 * mixed_tensor
        inputs_tensor = self.net.factor0 * inputs_tensor
        inputs_tensor = torch.maximum(torch.zeros_like(inputs_tensor), inputs_tensor) # ReLU, 禁止负值的出现(否则可能让 u 出现负数)
        inputs_tensor = inputs_tensor.to(device)
        if self.last_input == None:
            self.last_input = inputs_tensor
        
    
        
        th = self.cfg.steps
        for k in range(self.cfg.steps): # 进行 8 轮的动力学响应
            # if k + 1 <= th:
            #     real_input = (1 - (k + 1) / th) * self.last_input + (k + 1) / th * inputs_tensor
            # else:
            #     real_input = inputs_tensor
            real_input = inputs_tensor
            self.runner.set_input_directly(real_input)
            self.runner.execute(self.cfg.dt)
            pass
        
        self.last_input = inputs_tensor
        
        # 第四步, 根据 siamfc 的操作, 找到最匹配的尺度(响应值最大的尺度)
        max_index = torch.argmax(torch.max(responses.reshape(self.cfg.scale_num, -1), dim=1).values)
        response = responses[max_index] # 取出对应响应图
        cann_u = self.net.u[max_index]# 取出对应的 u
        sz = szs[max_index] # 取出对应的 sz
        
        # 第五步, 获得最终的响应图
        ## 按照 siamfc, 对 response 进行边缘抑制
        response = response - torch.min(response)
        response = response / torch.sum(response) + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * torch.tensor(self.hann_window, device=device)
        
        ## 为了使 response 能与 cann_u 进行混合, 需要进行归一化(不太确定是否需要)
        response = response / torch.max(response)
        cann_u = cann_u / torch.max(cann_u)
        cann_u_padded = torch.nn.functional.pad(cann_u, 
                                                (0, self.upscale_sz - self.cfg.len, 0, self.upscale_sz - self.cfg.len), 
                                                value=0) # 注: cann_u 为 271 * 271, response 为 272 * 272
        modified_response = self.net.mix_factor * response + (1. - self.net.mix_factor) * cann_u_padded # 混合
        
        # 第六步, 获得响应图中的最大值(软最大值)处
        if is_train == False:
            max_indice = torch.argmax(modified_response.reshape(-1))
            max_position4mix = torch.stack((max_indice // modified_response.shape[1], max_indice % modified_response.shape[1])).cpu()
        elif is_train == True:
            max_position4mix = self.criterion.get_center(modified_response.unsqueeze(0)).cpu()
        
        ## 这里找到 cann 的中心, 为之后进行循环位移做准备
        c_max_indice = torch.argmax(cann_u.detach().reshape(-1))
        c_max_position = torch.stack((c_max_indice // cann_u.shape[1], c_max_indice % cann_u.shape[1]))
        cann_center = c_max_position
        
        # 可视化
        if visualize:
            
            # self.ax[0, 0].imshow(responses_tensor[max_index].detach().cpu().numpy(), cmap='jet')
            # self.ax[0, 0].set_title('Responses')

            # self.ax[1, 0].imshow(movements_tensor[max_index].detach().cpu().numpy(), cmap='jet')
            # self.ax[1, 0].set_title('Movements')

            # self.ax[1, 1].imshow(mixed_tensor[max_index].detach().cpu().numpy(), cmap='jet')
            # self.ax[1, 1].set_title('Mixed')

            # self.ax[0, 1].imshow(inputs_tensor[max_index].detach().cpu().numpy(), cmap='jet')
            # self.ax[0, 1].set_title('Inputs')

            # self.ax[0, 2].imshow(self.net.u[max_index].detach().cpu().numpy(), cmap='jet')
            # self.ax[0, 2].set_title('CANN')
            
            # plt.tight_layout()
            # self.fig.show()
            
            # ops.show_response_in_img(img, img.shape, inputs_tensor[max_index].cpu().detach().numpy(),
            #     (sz, sz),
            #     self.center.cpu().detach().numpy(), visualize=True, border_value=(0, 0, 0),
            #     fig_n=4
            # )
            
            # ops.show_response_in_img(img, img.shape, self.net.u[max_index].cpu().detach().numpy(),
            #     (sz, sz),
            #     self.center.cpu().detach().numpy(), visualize=True, border_value=(0, 0, 0),
            #     fig_n=5
            # )
            
            pass
            
        # 第六步，获取对中心的预测
        disp_in_res = max_position4mix - self.constant_center
        disp_in_img = disp_in_res / self.cfg.len * sz
        self.center += disp_in_img.squeeze() # 更新中心位置
        self.net.roll(max_index, self.constant_center - cann_center.cpu())
        self.last_input = ops.roll(self.last_input, max_index, disp_in_res, 
                 self.cfg.scale_num, self.cfg.len)
        
        # 第七步，获取对尺寸的预测
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[max_index]
        self.target_sz *= scale
        self.x_sz *= scale
        
        # 第八步：从 chw/(cy, cy, h, w)格式转为 ltwh/(lx, ly, w, h) 形式的 box, 坐标系从 1 开始
        box = np.array([
            (self.center[1].detach().numpy() + 1) - (self.target_sz[1] - 1) / 2,
            (self.center[0].detach().numpy() + 1) - (self.target_sz[0] - 1) / 2,
            self.target_sz[1],
            self.target_sz[0]
        ])
        
        if video_saving:
            return box, modified_response.cpu().detach().numpy(), \
                inputs_tensor[max_index].cpu().detach().numpy(), c_max_position.cpu().detach().numpy(), sz
        else:
            return box, sz
          
    @torch.no_grad()    
    def update_like_siamfc(self, pre_img, img, is_train=False, visualize=False): 
        '''
        Functions: 和 siamfc 一致的追踪方式, 用于训练时的对比
        '''
        # 第一步：获得 responses
        self.siamfc.net.eval()
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0) # 转换为 numpy.ndarray, (3, 255, 255, 3)
        x = torch.from_numpy(x).to(
            device).permute(0, 3, 1, 2).float() # (3, 3, 255, 255)
         
        responses = self.get_resized_response(self.kernel, x)
        responses = responses.squeeze() # (3, 271, 271)
        responses = responses * torch.tensor(self.scale_penalty, device=device).unsqueeze(-1).unsqueeze(-1)



        max_index = torch.argmax(torch.max(responses.reshape(self.cfg.scale_num, -1), dim=1).values)
        response = responses[max_index]
        response = response - torch.min(response)
        response = response / torch.sum(response) + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * torch.tensor(self.hann_window, device=device)
        max_indice = torch.argmax(response.reshape(-1))
        max_position = torch.stack((max_indice // response.shape[1], max_indice % response.shape[1]))
        
        szs_in_img = self.upscale_sz * (self.cfg.total_stride / self.cfg.response_up) \
                                * (self.x_sz * self.scale_factors) / self.cfg.instance_sz
        sz = szs_in_img[max_index]    
        # ops.show_response_in_img(img, img.shape, inputs_tensor[max_index].cpu().detach().numpy(),
        #     (sz, sz),
        #     self.center, visualize=True, border_value=(0, 0, 0),
        #     fig_n=4
        # )
        # ops.show_response_in_img(img, img.shape, response.cpu().detach().numpy(),
        #     (calc.odd(sz), calc.odd(sz)),
        #     self.center, visualize=True, border_value=(0, 0, 0),
        #     fig_n=5
        # )
        
        # 第六步，获取对中心的预测
        disp_in_r = max_position.cpu() - (self.upscale_sz - 1) / 2
        disp_in_img = disp_in_r / self.upscale_sz * sz
        self.center += disp_in_img.squeeze().cpu().numpy()
        
        # 第七步，获取对尺寸的预测
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[max_index]
        self.target_sz *= scale
        self.x_sz *= scale
        
        # 第八步：返回 (lx, ly, w, h) 形式的 box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1],
            self.target_sz[0]
        ])
        
        return box, None
    
    @torch.enable_grad() # 以下是训练的时候用的
    def train_over(self, seqs, save_dir):
        '''
        Functions: 训练整个数据集
        '''
        save_dir = os.path.join(save_dir, self.formatted_date) # 设置保存路径
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.net.train()

        # 这里定义了数据集和数据加载器
        ## 训练数据集为 GOT-10K
        dataset = CANN_Pair(
            siamfc=self.siamfc,
            seqs=seqs,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=my_collate_fn,
            drop_last=True
        )
        
        # 这里定义了 loss 和参数的保存字典
        over_all_json_path = os.path.join(save_dir, 'loss_and_para.json')
        over_all_dict = {}
        
        # 整体训练
        for epoch in range(self.cfg.epoch_num):
            epoch_cann_ce2 = 0
            epoch_siamfc_ce2 = 0
            epoch_cann_ce = 0
            epoch_siamfc_ce = 0
            
            it_num = len(dataloader) # 限定每一个 epoch 的训练 batch 量, 太多吃不消
            
            # 遍历本次 epoch 每一个 batch
            for it, batch in enumerate(dataloader):
                if it > it_num - 1:
                    break
                
                # 训练该 batch, 并获得 loss
                cann_ce2, siamfc_ce2, cann_ce, siamfc_ce = self.train_batch(batch)
                epoch_cann_ce2 += cann_ce2
                epoch_siamfc_ce2 += cann_ce2
                epoch_cann_ce += cann_ce
                epoch_siamfc_ce += siamfc_ce
                
                # 输出相关信息
                print('Epoch: {} [{}/{}] \nCANN AVG CE2: {:.2f}; SiamFC AVG CE2: {:.2f}\nCANN AVG CE: {:.2f}; SiamFC AVG CE: {:.2f}'.format(
                    epoch + 1, it + 1, len(dataloader), 
                    cann_ce2, siamfc_ce2,
                    cann_ce, siamfc_ce))
                print("Value of a is ", np.array2string(self.net.a.detach().cpu().numpy(),precision=5, floatmode='fixed'))
                print("Value of A is ", np.array2string(self.net.A.detach().cpu().numpy(),precision=5, floatmode='fixed'))
                print("Value of k is ", np.array2string(self.net.k.detach().cpu().numpy(),precision=5, floatmode='fixed'))
                print("Value of tau is ", np.array2string(self.net.tau.detach().cpu().numpy(),precision=5, floatmode='fixed'))
                print("Value of factor for inputs is ", np.array2string(self.net.factor0.detach().cpu().numpy(),precision=5, floatmode='fixed'))
                print("Value of factor for movements is", np.array2string(self.net.factor1.detach().cpu().numpy(),precision=5, floatmode='fixed'))
                print("Value of factor for mixeds is ", np.array2string(self.net.factor2.detach().cpu().numpy(),precision=5, floatmode='fixed'))
                print("Value of factor for total mixeds is ", np.array2string(self.net.mix_factor.detach().cpu().numpy(),precision=5, floatmode='fixed'))
                print("")
                sys.stdout.flush()

                # 保存 loss 和参数信息
                over_all_dict[(epoch + 1, it + 1)] = {
                    'cann_ce2': cann_ce2,
                    'siamfc_ce2': siamfc_ce2,
                    'cann_ce': cann_ce,
                    'siamfc_ce': siamfc_ce,
                    'a': float(self.net.a.detach().cpu().numpy()),
                    'A': float(self.net.A.detach().cpu().numpy()),
                    'k': float(self.net.k.detach().cpu().numpy()),
                    'tau': float(self.net.tau.detach().cpu().numpy()),
                    'factor0': float(self.net.factor0.detach().cpu().numpy()),
                    'factor1': float(self.net.factor1.detach().cpu().numpy()),
                    'factor2': float(self.net.factor2.detach().cpu().numpy()),
                    'mix_factor': float(self.net.mix_factor.detach().cpu().numpy())
                }
            
            # 保存当前 epoch 的 loss 和参数信息
            over_all_dict[(epoch + 1, 0)] = {
                'cann_ce2': epoch_cann_ce2 / it_num,
                'siamfc_ce2': epoch_siamfc_ce2 / it_num,
                'cann_ce': epoch_cann_ce / it_num,
                'siamfc_ce': epoch_siamfc_ce / it_num,
                'a': float(self.net.a.detach().cpu().numpy()),
                'A': float(self.net.A.detach().cpu().numpy()),
                'k': float(self.net.k.detach().cpu().numpy()),
                'tau': float(self.net.tau.detach().cpu().numpy()),
                'factor0': float(self.net.factor0.detach().cpu().numpy()),
                'factor1': float(self.net.factor1.detach().cpu().numpy()),
                'factor2': float(self.net.factor2.detach().cpu().numpy()),
                'mix_factor': float(self.net.mix_factor.detach().cpu().numpy())
            }
            
            # 进行学习率调整
            self.lr_scheduler.step()
                   
            # 保存网络参数
            net_path = os.path.join(
                save_dir, 'siamfc_cann_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
            
            # 保存字典
            with open(over_all_json_path, 'w') as f:
                json.dump({str(k): over_all_dict[k] for k in over_all_dict}, f)
    
    def train_batch(self, batch):
        '''
        Functions: 训练当前 batch 内的 batch_size 个序列
        '''
        self.net.train(True)
        self.siamfc.net.eval()
        seq_imgs_path_list, seq_annos_list, seq_lens_list = batch
        # batch[0]: 序列图像路径 list, 4 个 (seq_len)
        # batch[1]: 标注框框 list, 4 个 (seq_len, 4)
        # batch[2]: 序列长度 list, 4 个 (1)
        
        with torch.set_grad_enabled(True):
            
            cann_ce2 = 0
            siamfc_ce2 = 0
            cann_ce = 0
            siamfc_ce = 0
            self.optimizer.zero_grad()
            
            for b in range(self.cfg.batch_size):
                print("batch_num:", b, end=";")
                # 第一步: 取出数据
                seq_imgs_path, seq_annos, seq_len = seq_imgs_path_list[b], seq_annos_list[b], seq_lens_list[b] 
                # 第二步: 进行全序列追踪, 并获得 loss
                b_cann_ce2, b_siamfc_ce2, b_cann_ce, b_siamfc_ce = self.track(
                    seq_imgs_path, seq_annos[0], seq_annos, 'train', is_train=True, visualize=True
                ) # loss 是 ce ** 2
                
                b_cann_ce2 = b_cann_ce2 / self.cfg.batch_size
                b_cann_ce2.backward()
                
                cann_ce2 += b_cann_ce2
                siamfc_ce2 += b_siamfc_ce2
                cann_ce += b_cann_ce
                siamfc_ce += b_siamfc_ce
                
            cann_ce2 = cann_ce2 
            siamfc_ce2 = siamfc_ce2 / self.cfg.batch_size
            cann_ce = cann_ce / self.cfg.batch_size
            siamfc_ce = siamfc_ce / self.cfg.batch_size
            
            # 在 self.track 中已经反向传播, 这里直接梯度下降
            self.optimizer.step()
            
            # 输出梯度信息
            for name, param in self.net.named_parameters():
                print(f"Parameter name: {name}")
                print(f"Gradient: {param.grad}")
        
        return cann_ce2.item(), siamfc_ce2.item(), cann_ce, siamfc_ce
    
    
    def get_resized_response(self, kernel, instance):
        '''
        Functions: 返回 272 * 272 的响应图
        '''
        instance_features = self.siamfc.net.backbone(instance) # 获得 (bs, 256, 6, 6) 的搜索区域特征图
        responses = self.siamfc.net.head(kernel, instance_features) # 获得 (bs, 1, 17, 17) 的响应图
        responses = ops.upsample(responses, self.cfg.response_sz,
                                    self.upscale_sz) # 获得 (bs, 1, 272, 272) 的响应图
        return responses
    
    def track_comparison(self, img_files, box, annos, seq_name, save_dir):
        '''
        Functions: 对比 siamfc 和 cann 的追踪效果
        '''
        # 初始化一些参数
        pre_img = None # 用于保存上一帧图像
        video_save = ComparisonSave(save_dir)
        video_save.init(seq_name)
        
        # 遍历每一帧图像, 并追踪
        for f, img_file in enumerate(img_files): 
            img = ops.read_image(img_file) # 读入图像，格式为: RGB, [0, 255], shape=(h, w, 3)
            
            if f == 0: # 第一帧，抽取模板图像，生成卷积核
                self.init(img, box) 
                self.siamfc.init(img, box)
                box_cann = box
                box_siam = box
            else: # 后续帧：生成响应图，生成标注框
                if pre_img.shape != img.shape: # 有些训练数据里面有问题，需要跳过
                    break 
                
                search_center4siam, _ = ops.get_center_sz(box_siam)
                search_center4cann, _ = ops.get_center_sz(box_cann)
                
                box_cann, res_cann, input_cann, peak_cann, sz_cann = self.update(pre_img, img, is_train=False, visualize=True,
                                                                                video_saving=True) # 更新标注框
                peak_cann = np.array([peak_cann[1], peak_cann[0]])
                box_siam, res_siam, sz_siam = self.siamfc.update(img, video_saving=True)
        
                pre_in_img4siam, _ = ops.get_center_sz(box_siam)
                pre_in_img4cann, _ = ops.get_center_sz(box_cann)
                gt_in_img, _ = ops.get_center_sz(annos[f])
                
                gt_disp_in_img4siam = search_center4siam - gt_in_img
                gt_disp_in_img4cann = search_center4cann - gt_in_img
                gt_disp_in_res4siam = gt_disp_in_img4siam * self.cfg.len / sz_siam
                gt_disp_in_res4cann = gt_disp_in_img4cann * self.cfg.len / sz_cann
                pre_disp_in_img4siam = pre_in_img4siam - search_center4siam
                pre_disp_in_img4cann = pre_in_img4cann - search_center4cann
                pre_disp_in_res4siam = pre_disp_in_img4siam * self.cfg.len / sz_siam
                pre_disp_in_res4cann = pre_disp_in_img4cann * self.cfg.len / sz_cann
                
                gt_in_res4siam = np.array(res_siam.shape) // 2 + gt_disp_in_res4siam
                gt_in_res4cann = np.array(res_cann.shape) // 2 + gt_disp_in_res4cann
                pre_in_res4siam = np.array(res_siam.shape) // 2 + pre_disp_in_res4siam
                pre_in_res4cann = np.array(res_cann.shape) // 2 + pre_disp_in_res4cann
                
                
                img12 = ops.show_image(img, boxes=[box_siam, annos[f, :], box_cann],
                                       colors=[(0, 0, 255), (0, 255, 0), (255, 0, 0)], 
                                       fig_n=1, visualize=True)
                img1 = ops.show_image(img, [box_siam, annos[f, :]], 
                                      colors=[(0, 0, 255), (0, 255, 0)],
                                      visualize=False)
                img1 = ops.show_response_in_img(img1, img1.shape, res_siam, (sz_siam, sz_siam), search_center4siam, 
                                                fig_n=2, visualize=True)
                img2 = ops.show_image(img, [box_cann, annos[f, :]], 
                                      colors=[(255, 0, 0), (0, 255, 0)],
                                      visualize=False)
                img2 = ops.show_response_in_img(img2, img2.shape, res_cann, (sz_cann, sz_cann), search_center4cann, 
                                                fig_n=3, visualize=True)
                res1 = ops.show_response(res_siam, pre_in_res4siam, gt_in_res4siam, 
                                         colors=[(255, 255, 255), (0, 0, 255), (0, 255, 0)],
                                         fig_n=4, visualize=True)
                res2 = ops.show_response(input_cann, peak_cann, gt_in_res4cann, 
                                         colors=[(255, 255, 255), (255, 0, 0), (0, 255, 0)],
                                         fig_n=5, visualize=True)
                
                video_save.append(img12, img1, img2, res1, res2)
                
            pre_img = img
        
        video_save.save()
    