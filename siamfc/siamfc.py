from __future__ import absolute_import, division, print_function

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
import torch.optim as optim
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from got10k.trackers import Tracker
from utils import ops, ops_torch, video_save

from .backbones import AlexNetV1
from .datasets import Pair
from .heads import SiamFC
from .losses import BalancedLoss, _BalancedLoss
from .transforms import SiamFCTransforms

__all__ = ['TrackerSiamFC']


st1, _t1 = 0, 0
st2, _t2 = 0, 0
st3, _t3 = 0, 0


# Net: 整体网络结构
class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        # z, x 为样本和搜索区域，最终输出为二者通过 Alexnet 得到的特征图经过卷积后的响应图
        # 本项目中, z.shape=(bs, 3, 127, 127), x.shape=(bs, 3, 255, 255)
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, failure_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True) # 没啥用的初始化，父类只是定义方法
        self.cfg = self.parse_args(**kwargs) # 设置参数
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        now = datetime.now()
        self.formatted_date = now.strftime("%Y_%m_%d_%H_%M_%S") # 获取当前时间，用于保存模型
        # 是否进行失败记录
        self.failure_path = failure_path
        if self.failure_path is not None:
            self.failure_save = video_save.FailureSave(self.failure_path)
            
        # 设置模型运行的设备（GPU）
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # 定义好模型
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net) # 初始化整体网络参数
        
        # 加载预训练模型（或者 checkpoint）
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # 设置损失函数
        self.criterion = BalancedLoss()
        self._criterion = _BalancedLoss()

        # 设置优化器
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay, # 权重衰减：防止过拟合
            momentum=self.cfg.momentum) # 动量参数：加速收敛
        
        # 设置学习器
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num) # (终止/起始)^(1/epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma) # 指数衰减的学习率, lr = lr_0 * (gamma^epoch)
    
    @torch.no_grad()
    def init(self, img, box):
        # 设置成评估模式，因为训练的时候其实压根没用这个函数（
        self.net.eval()

        # 输入的 annotations 的格式是 (x, y, x_w, y_h), 也就是左上角以及高和宽
        # 但我们要转化成的 box 的格式是 (c_y, c_x, h, w), 也就是中心点以及高和宽
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:] # 获得第一帧的中心点 (c_y, c_x) 以及尺寸 (h, w)

        # 17 * 17 的响应图最终会被扩大为 272 * 272，以方便寻找最大值点
        ## self.cfg.response_up = 16, self.cfg.response_sz = 17 
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz 
        # 汉宁窗以余弦递减抑制边缘区域的权值，这个是最后拿来加的时候用的
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window_sigmoid = self.hann_window
        self.hann_window_sigmoid = 1. / (1. + np.exp(-self.hann_window_sigmoid))
        
        self.hann_window /= self.hann_window.sum() # 分布归一化，得到的值小于 1 / (272 * 272)
        self.hann_window_sigmoid /= self.hann_window_sigmoid.sum() # 分布归一化，得到的值小于 1 / (272 * 272)
        
        # 缩放因子，意义需要在下文再解释
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num) # 0.964, 1.000, 1.0375 

        # 获得裁剪全图时，样本图像 z 的大小: 1.5\sqrt{dy * dy} 和搜索区域 x 的大小 
        context = self.cfg.context * np.sum(self.target_sz) # 0.5 * (d_y, d_x)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context)) # 1.5 * (d_y, d_x), 然后得到 1.5 * \sqrt{d_y * d_x}
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz # 按比例缩放，系数即为 255 / 127
        
        # 然后根据上面的尺寸，裁剪出 self.z_sz 大小的图像，然后填充并缩放成 127 * 127
        self.avg_color = np.mean(img, axis=(0, 1)) # 获得三通道均值，用于归一化 (*, *, *)
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)
        
        # 样本图进行一些维度处理, (h, w, c) -> (bs, c, h, w)
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        
        # 根据样本图获得卷积核 kernel, size = (1, 256, 6, 6) 
        self.kernel = self.net.backbone(z)
        pass
    @torch.no_grad()
    def update(self, img, video_saving=False): ##### 就是输入匹配帧，然后根据模板图像匹配，最后生成标注框
        # 设置成评估模式，因为训练的时候其实压根没用这个函数（
        self.net.eval()

        # 根据上一帧得到的图片中心，裁剪出搜索
        ## 首先，图像为 (h, w, 3) 
        ## 其次，在中心点 (y, x) 裁剪出 x_sz * f 的图像, f 为缩放因子 (0.964, 1.000, 1.0375), 目的是为了得到三个尺度的样本图
        ## 以调整 bounding box 的大小
        ## 最后裁剪后，输出的图像都是 255 * 255 的大小
        # global st1, _t1, st2, _t2, st3, _t3
        
        # self.avg_color = (0, 0, 0)
        
        # st1 = time.time()
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        # _t1 += time.time() - st1
        
        # st2 = time.time()
        # x_ = [ops_torch.crop_and_resize_optimized(
        #     img, self.center, self.x_sz * f,
        #     out_size=self.cfg.instance_sz,
        #     border_value=self.avg_color) for f in self.scale_factors]
        # _t2 += time.time() - st2
        
        

        
        x = np.stack(x, axis=0) # 转换为 numpy.ndarray, (3, 255, 255, 3)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float() # (3, 3, 255, 255) 
        
        
        # 得到搜索区域提取得到的特征图
        responses = self.get_resized_response(x)
        responses = responses.squeeze(1).cpu().numpy() # 压缩 1 维度，获得 (3, 272, 272)
        
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty # 权值的尺度惩罚 0.9745，因为缩小了
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty # 权值的尺度惩罚 0.9745，因为放大了

        # peak scale，比较所有采样图的峰值处谁最高，就要 (0, 1, 2) 中的哪一张 
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location，终于找到采样图了 (272, 272)
        response = responses[scale_id]
        origin_response = responses[scale_id]
        response -= response.min() # 变成正值
        response /= response.sum() + 1e-16 # 分布归一化
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * 1.00 * self.hann_window # (1-a) * M + a * H，一定程度地抑制边缘(但程度不大......)，仍分布归一化
        loc = np.unravel_index(response.argmax(), response.shape) # 找到峰值坐标并变成元组

        # 绘制采样图，查看情况
        
        # response_img, rp_mmin, rp_mmax = ops.show_response(response)
        
        # 绘制得到的采样图的原始数据
        '''
        plt.imshow(origin_response, cmap='viridis', interpolation='nearest')
        mmin = np.min(origin_response)
        mmax = np.max(origin_response)
        avg = np.average(origin_response)
        plt.title("mmin: " + str(mmin) + "\nmmax: " + str(mmax) + "\navg: " + str(avg))
        plt.colorbar()  # 添加颜色条
        plt.show()
        '''
        # 绘制对整张图进行卷积得到的响应
        '''
        self.response_for_whole(img)
        '''  
        # 把采样图尝试映射回原图大小，看看效果
        ## 首先采样图已经被缩放为 272*272，所以要首先映射回 272 * 8 / 16 = 136 得到样本图大小
        ## 然后再 136 * (x_xz * scale_factor / instance_sz) 得到原图像大小
        ## 最后把热度图嵌入原图像（也就是当前的中心），并设置其它地方为 0 
        
        # size_in_image = np.asarray(response.shape) * self.cfg.total_stride / self.cfg.response_up \
        #     * self.x_sz * self.scale_factors[scale_id] / self.cfg.instance_sz
        # size_in_image = np.round(size_in_image).astype(int)
        # for i in range(0, len(size_in_image)):
        #     if size_in_image[i] % 2 == 0:
        #         size_in_image[i] += 1
        # assert size_in_image[0] % 2 == 1 and size_in_image[1] % 2 == 1 # 保证是奇数，这样才好有中心
        # # response_in_img = ops.show_response_in_img(img, np.asarray(img.shape), response, size_in_image, self.center, 
        #                                            border_value=self.avg_color)
        
        
        # 峰值点映射回原图的位置，并更新位置
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2 # 响应图中，峰值位置减去中心位置，就是偏移量
        disp_in_instance = disp_in_response * self.cfg.total_stride / self.cfg.response_up # 由响应图位移倒推样本图位移
        disp_in_image = disp_in_instance * (self.x_sz * self.scale_factors[scale_id]) / \
            self.cfg.instance_sz # 由样本图倒推原图像位移
        sz_in_img = self.upscale_sz * (self.cfg.total_stride / self.cfg.response_up) \
                                    * (self.x_sz * self.scale_factors[scale_id] / self.cfg.instance_sz)
        self.center += disp_in_image # 得到新的中心

        # 以 0.59 的学习率更新目标大小
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # 返回 (lx, ly, w, h) 形式的 box
        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        if self.failure_path is not None:
            self.failure_save.rps.append(response_img)
            self.failure_save.rps_mmin.append(rp_mmin)
            self.failure_save.rps_mmax.append(rp_mmax)
            self.failure_save.imgs_rp.append(response_in_img)
        
        if video_saving:
            return box, response, sz_in_img
        else:
            return box, sz_in_img
    
    @torch.no_grad()
    def update_sigmoid(self, img): ##### 就是输入匹配帧，然后根据模板图像匹配，最后生成标注框
        # 设置成评估模式，因为训练的时候其实压根没用这个函数（
        self.net.eval()

        # 根据上一帧得到的图片中心，裁剪出搜索
        ## 首先，图像为 (h, w, 3) 
        ## 其次，在中心点 (y, x) 裁剪出 x_sz * f 的图像, f 为缩放因子 (0.964, 1.000, 1.0375), 目的是为了得到三个尺度的样本图
        ## 以调整 bounding box 的大小
        ## 最后裁剪后，输出的图像都是 255 * 255 的大小
        # global st1, _t1, st2, _t2, st3, _t3
        
        # self.avg_color = (0, 0, 0)
        
        # st1 = time.time()
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        # _t1 += time.time() - st1
        
        # st2 = time.time()
        # x_ = [ops_torch.crop_and_resize_optimized(
        #     img, self.center, self.x_sz * f,
        #     out_size=self.cfg.instance_sz,
        #     border_value=self.avg_color) for f in self.scale_factors]
        # _t2 += time.time() - st2
        
        

        
        x = np.stack(x, axis=0) # 转换为 numpy.ndarray, (3, 255, 255, 3)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float() # (3, 3, 255, 255) 
        
        
        # 得到搜索区域提取得到的特征图
        responses = self.get_resized_response(x)
        responses = responses.squeeze(1).cpu().numpy() # 压缩 1 维度，获得 (3, 272, 272)
        
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty # 权值的尺度惩罚 0.9745，因为缩小了
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty # 权值的尺度惩罚 0.9745，因为放大了

        # peak scale，比较所有采样图的峰值处谁最高，就要 (0, 1, 2) 中的哪一张 
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location，终于找到采样图了 (272, 272)
        response = responses[scale_id]
        origin_response = responses[scale_id]
        response = 1. / (1. + np.exp(-response))
        response /= response.sum() + 1e-16 # 分布归一化
        
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window_sigmoid # (1-a) * M + a * H，一定程度地抑制边缘(但程度不大......)，仍分布归一化
        loc = np.unravel_index(response.argmax(), response.shape) # 找到峰值坐标并变成元组

        # 绘制采样图，查看情况
        
        response_img, rp_mmin, rp_mmax = ops.show_response(response)
        
        # 绘制得到的采样图的原始数据
        '''
        plt.imshow(origin_response, cmap='viridis', interpolation='nearest')
        mmin = np.min(origin_response)
        mmax = np.max(origin_response)
        avg = np.average(origin_response)
        plt.title("mmin: " + str(mmin) + "\nmmax: " + str(mmax) + "\navg: " + str(avg))
        plt.colorbar()  # 添加颜色条
        plt.show()
        '''
        # 绘制对整张图进行卷积得到的响应
        '''
        self.response_for_whole(img)
        '''  
        # 把采样图尝试映射回原图大小，看看效果
        ## 首先采样图已经被缩放为 272*272，所以要首先映射回 272 * 8 / 16 = 136 得到样本图大小
        ## 然后再 136 * (x_xz * scale_factor / instance_sz) 得到原图像大小
        ## 最后把热度图嵌入原图像（也就是当前的中心），并设置其它地方为 0 
        
        # size_in_image = np.asarray(response.shape) * self.cfg.total_stride / self.cfg.response_up \
        #     * self.x_sz * self.scale_factors[scale_id] / self.cfg.instance_sz
        # size_in_image = np.round(size_in_image).astype(int)
        # for i in range(0, len(size_in_image)):
        #     if size_in_image[i] % 2 == 0:
        #         size_in_image[i] += 1
        # assert size_in_image[0] % 2 == 1 and size_in_image[1] % 2 == 1 # 保证是奇数，这样才好有中心
        # # response_in_img = ops.show_response_in_img(img, np.asarray(img.shape), response, size_in_image, self.center, 
        #                                            border_value=self.avg_color)
        
        
        # 峰值点映射回原图的位置，并更新位置
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2 # 响应图中，峰值位置减去中心位置，就是偏移量
        disp_in_instance = disp_in_response * self.cfg.total_stride / self.cfg.response_up # 由响应图位移倒推样本图位移
        disp_in_image = disp_in_instance * (self.x_sz * self.scale_factors[scale_id]) / \
            self.cfg.instance_sz # 由样本图倒推原图像位移
        self.center += disp_in_image # 得到新的中心

        # 以 0.59 的学习率更新目标大小
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # 返回 (lx, ly, w, h) 形式的 box
        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        if self.failure_path is not None:
            self.failure_save.rps.append(response_img)
            self.failure_save.rps_mmin.append(rp_mmin)
            self.failure_save.rps_mmax.append(rp_mmax)
            self.failure_save.imgs_rp.append(response_in_img)
        
        return box
 
    
    def track(self, img_files, box, anno, seq_name='',
              visualize=False, video_save=False, is_record_delta=False):
        # [img_paths], [annotations], 追踪一个视频序列
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4)) # 注意，标注框的格式为 (c_y, c_x, d_y, d_x), 但是输入的 box(annotations) 是 (x, y, w, h)
        boxes[0] = box # 初始化第一帧的标注框
        times = np.zeros(frame_num)
        imgs = [] # 用于保存结果图像
        
        if self.failure_path is not None:
            self.failure_save.init(seq_name)
        
        delta = []
        
        # 遍历每一帧图像
        for f, img_file in enumerate(img_files): 
            img = ops.read_image(img_file) # 读入图像，并转为 RGB 格式，且形状为 (h, w, 3)

            begin = time.time()
            if f == 0: # 第一帧，抽取模板图像，生成卷积核
                self.init(img, box) 
                sz = 1
            else: # 后续帧：生成响应图，生成标注框
                boxes[f], sz = self.update(img) # Fix: Remove extra indexing
                if self.failure_path is not None:
                    self.failure_save.annotations.append(anno[f, :])
                    self.failure_save.boxes.append(boxes[f, :])
                    self.failure_save.calc_IoU()
                
                search_center, _ = ops.get_center_sz(boxes[f - 1])
                gt_center, _ = ops.get_center_sz(anno[f])
                gt_disp_in_img = gt_center - search_center
                gt_disp_in_res = gt_disp_in_img / sz * self.cfg.len
                
                delta.append(np.linalg.norm(gt_disp_in_res))
                
            times[f] = time.time() - begin

            
            
            
            if visualize:
                img = ops.show_image(img, [boxes[f, :], anno[f, :]]) # Fix: Remove extra indexing
                if video_save:
                    imgs.append(img)
                if self.failure_path is not None and f != 0:
                    self.failure_save.imgs.append(img)


        if self.failure_path is not None:
            self.failure_save.save()
        
        return boxes, times, imgs, delta # 返回每一帧的标注框以及每一帧图像的用时
    
    def train_step(self, batch, backward=True):
        # set network mode(实际上是重复设置)
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference, shape=(8, 1, 15, 15)，我就纳闷了，训练用 15 * 15，推理用 17 * 17，干啥呢？？
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size()) # 15 * 15 中心曼哈顿距离为 2 以内的为正样本，其余为负样本
            loss = self.criterion(responses, labels) # 然后就算 loss
            
            _responses = torch.sigmoid(responses) # 对 responses 进行 sigmoid 化
            _loss = self._criterion(_responses, labels) # 计算交叉熵损失
            
            print(loss.cpu().detach().item(), _loss.cpu().detach().item())
            
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # 更改保存路径
        save_dir = os.path.join(save_dir, self.formatted_date)
        
        # set to train mode(梯度不冻结，与 self.net.eval() 对应)
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 定义数据集类
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        ## seqs 就是用于加载 GOT10K 的数据集类，不过还要和 transforms 整合
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)
        
        # 迭代器，用到再说
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        
        # loop over epochs（训练阶段：迭代轮数）
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            ## 长度为 [9335 / 8] 的迭代器 
            for it, batch in enumerate(dataloader):
                # batch[0]: 样本图像, (8, 3, 127, 127)
                # batch[1]: 搜索区域, (8, 3, 239, 239)
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
    
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels

    def get_resized_response(self, instance, exemplar=None):
        if exemplar is not None:
            exemplar = exemplar.to(self.device)
            kernel = self.net.backbone(exemplar)
        else:
            kernel = self.kernel
        instance = instance.to(self.device)
        instance_features = self.net.backbone(instance)
        responses = self.net.head(kernel, instance_features)
        
        # import matplotlib.pyplot as plt

        # # Plot the response as a heatmap
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # mat1 = responses[0].squeeze(0).cpu().numpy()
        # mat2 = 1 / (1 + np.exp(-mat1))
        # axs[0].imshow(mat1, cmap='jet')
        # axs[0].set_title('non-sigmoided response')
        # axs[0].axis('off')
        # axs[1].imshow(mat2, cmap='jet')
        # axs[1].set_title('sigmoided response')
        # axs[1].axis('off')
        # plt.show()
        
        responses = ops.upsample(responses, self.cfg.response_sz,
                                    self.cfg.response_sz * self.cfg.response_up)
        return responses

    def filt_peak(self, response, instance_center, 
                  tolerance_height, tolerance_dis):
        smoothing_sigma = 1
        min_distance = 10
        threshold_abs = 0.1
        pad_width = 20
        # 步骤 0: 填充，使得边缘的局部最值能被找到
        response = np.pad(response, pad_width=pad_width, mode='constant')
        instance_center = instance_center + pad_width
        
        # 步骤 1: 平滑处理
        smoothed_response = gaussian_filter(response, sigma=smoothing_sigma)
        
        # 步骤 2: 寻找局部最大值
        peaks = peak_local_max(smoothed_response, min_distance=min_distance, threshold_abs=threshold_abs, num_peaks=np.inf)
        
        # 检查是否找到了足够的山峰
        if len(peaks) < 1:
            return False
        
        # 步骤 4: 获取最高和次高山峰的峰值(因为归一化，最高的肯定是1)
        peak_values = smoothed_response[peaks[:, 0], peaks[:, 1]]
        valid_peaks = peaks[peak_values >= tolerance_height, :]  # 获取大于 tolerance 的峰值
        
        # 步骤 5: 至少有两个峰
        if len(valid_peaks) < 2:
            return False
        
        # 步骤 6: 检查中心点是否在某个峰附近        
        ## 计算给定点到所有有效山峰的距离
        distances = cdist(valid_peaks, np.array([instance_center]))
        nearest_peak_distance = np.min(distances)
        if nearest_peak_distance > tolerance_dis:
            return False

        return True

    def response_for_whole(self, img):
        x = torch.from_numpy(img).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
        x = self.net.backbone(x) # (1, 256, *, *)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze().cpu().numpy()
        responses = cv2.resize(responses, 
                               (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_CUBIC)
        ops.show_whole_img_response(img, responses)
    
    def _failure_save(self, imgs, imgs_rp, rp, boxes, annos, indexes, save_dir):
        pass 