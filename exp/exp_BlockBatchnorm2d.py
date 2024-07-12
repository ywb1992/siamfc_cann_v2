import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from tqdm import tqdm

work_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # 工作路径，用于存放代码
parent_path = os.path.abspath(os.path.join(work_path, '..')) # 上一级路径
sys.path.append(work_path) # 加载工作路径

torch.manual_seed(0)

class BlockBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, block_num, block_size, ori_channels):
        super(BlockBatchNorm2d, self).__init__(block_num * ori_channels)
        self.block_num = block_num
        self.block_size = block_size
        self.ori_channels = ori_channels
    

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.permute(1, 0, 2, 3) # (c, n, h, w)
        x = x.reshape(c * self.block_num, self.block_size, h, w) # (c * block_num, block_size, h, w)
        x = x.permute(1, 0, 2, 3) # (block_size, c * block_num, h, w)
        # 应用 batchnorm
        x = super(nn.BatchNorm2d, self).forward(x) # 对 block_size * c 个通道进行 batchnorm
        # 恢复原始维度
        x = x.permute(1, 0, 2, 3) # (c * block_num, block_size, h, w)
        x = x.reshape(c, n, h, w) # (c, n, h, w)
        x = x.permute(1, 0, 2, 3) # (n, c, h, w)
        return x

class _BlockBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, block_num, block_size, ori_channels):
        super(_BlockBatchNorm2d, self).__init__(
            block_num * ori_channels, eps=1e-6, momentum=0.05
        )
        self.block_num = block_num
        self.block_size = block_size
        self.ori_channels = ori_channels
    

    def forward(self, x: torch.Tensor):
        n, c, h, w = x.shape
        # b = self.block_size
        # x_norm = []
        # for i in range(self.block_num):
        #     x_norm.append(super(_BlockBatchNorm2d, self).forward(
        #         x[i * b : (i + 1) * b])
        #     )
        # x_norm = torch.cat(x_norm, dim=0)
        # return x_norm
        x = x.permute(1, 0, 2, 3) # (c, n, h, w)
        x = x.reshape(c * self.block_num, self.block_size, h, w) # (c * block_num, block_size, h, w)
        x = x.permute(1, 0, 2, 3) # (block_size, c * block_num, h, w)
        # 应用 batchnorm
        x = super(_BlockBatchNorm2d, self).forward(x) # 对 c * block_num 个通道进行 batchnorm
        # 恢复原始维度
        x = x.permute(1, 0, 2, 3) # (c * block_num, block_size, h, w)
        x = x.reshape(c, n, h, w) # (c, n, h, w)
        x = x.permute(1, 0, 2, 3) # (n, c, h, w)
        return x

class _BlockBatchNorm3d(nn.BatchNorm3d):
    def __init__(self, block_num, block_size, ori_channels):
        super(_BlockBatchNorm3d, self).__init__(
            ori_channels, eps=1e-6, momentum=0.05
        )
        self.block_num = block_num
        self.block_size = block_size
        self.ori_channels = ori_channels
    

    def forward(self, x: torch.Tensor):
        n, c, h, w = x.shape
        x = x.permute(1, 0, 2, 3) # (c, n, h, w)
        x = x.reshape(c, self.block_num, self.block_size, h, w) # (c, block_num, block_size, h, w)
        x = x.permute(2, 0, 1, 3, 4) # (block_size, c, block_num, h, w)
        # 应用 batchnorm
        x = super(_BlockBatchNorm3d, self).forward(x) # 对 c 个通道进行 batchnorm
        # 恢复原始维度
        x = x.permute(1, 2, 0, 3, 4) # (c, block_num, block_size, h, w)
        x = x.reshape(c, n, h, w) # (c, n, h, w)
        x = x.permute(1, 0, 2, 3) # (n, c, h, w)
        return x


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)

class _AlexNet(nn.Module):
    
    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2[0](x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

# 一般所使用的 AlexNet 网络
class AlexNetV1(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight)
                if m.bias is not None:
                    init.normal_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight)
                if m.bias is not None:
                    init.normal_(m.bias, 0)
            elif isinstance(m, _BatchNorm2d):
                init.normal_(m.weight, 1)
                init.normal_(m.bias, 0)   

# 搭载了 block 版本的 AlexNet
class BlockAlexNetV1(_AlexNet):
    output_stride = 8

    def __init__(self, ori_alexnet: AlexNetV1, block_num, block_size):
        super(BlockAlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BlockBatchNorm2d(block_num, block_size, 96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BlockBatchNorm2d(block_num, block_size, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BlockBatchNorm2d(block_num, block_size, 384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BlockBatchNorm2d(block_num, block_size, 384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))
        self._initialize_weights(ori_alexnet, block_num, block_size)
    
    def _initialize_weights(self, ori_alexnet: AlexNetV1, block_num, block_size):
        for m, ori_m in zip(self.modules(), ori_alexnet.modules()):
            if isinstance(m, nn.Conv2d):
                m.weight = nn.Parameter(ori_m.weight.clone())
                m.bias = nn.Parameter(ori_m.bias.clone())
            elif isinstance(m, _BlockBatchNorm2d):
                num_channels = ori_m.num_features
                m.weight = nn.Parameter(ori_m.weight.clone().unsqueeze(-1).broadcast_to(num_channels,
                                                                                         block_num).reshape(-1))
                m.bias = nn.Parameter(ori_m.bias.clone().unsqueeze(-1).broadcast_to(num_channels,
                                                                                         block_num).reshape(-1))
            elif isinstance(m, nn.Linear):
                m.weight = nn.Parameter(ori_m.weight.clone())
                m.bias = nn.Parameter(ori_m.bias.clone())

# 搭载了 3d-block 版本的 AlexNet
class BlockAlexNetV1_3D(_AlexNet):
    output_stride = 8

    def __init__(self, ori_alexnet: AlexNetV1, block_num, block_size):
        super(BlockAlexNetV1_3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BlockBatchNorm3d(block_num, block_size, 96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BlockBatchNorm3d(block_num, block_size, 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BlockBatchNorm3d(block_num, block_size, 384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BlockBatchNorm3d(block_num, block_size, 384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))
        self._initialize_weights(ori_alexnet, block_num, block_size)
    
    def _initialize_weights(self, ori_alexnet: AlexNetV1, block_num, block_size):
        for m, ori_m in zip(self.modules(), ori_alexnet.modules()):
            if isinstance(m, nn.Conv2d):
                m.weight = nn.Parameter(ori_m.weight.clone())
                m.bias = nn.Parameter(ori_m.bias.clone())
            elif isinstance(m, _BlockBatchNorm3d):
                num_channels = ori_m.num_features
                m.weight = nn.Parameter(ori_m.weight.clone().broadcast_to(num_channels))
                m.bias = nn.Parameter(ori_m.bias.clone().broadcast_to(num_channels))
            elif isinstance(m, nn.Linear):
                m.weight = nn.Parameter(ori_m.weight.clone())
                m.bias = nn.Parameter(ori_m.bias.clone())



def compare_block_bn_and_explicit_bn():
    batch_size = 4
    num_channels = 2
    height = 1
    width = 1
    block_size = 2
    block_num = batch_size // block_size

    # 创建 BlockBatchNorm2d 实例
    block_bn = BlockBatchNorm2d(block_num=block_num, block_size=block_size, ori_channels=num_channels)
    
    # 创建 BatchNorm2d 实例
    batch_norm = nn.BatchNorm2d(num_channels)
    # 随机设置 BatchNorm2d 的初始权重
    batch_norm.weight.data = torch.randn_like(batch_norm.weight.data)
    batch_norm.bias.data = torch.randn_like(batch_norm.bias.data)
    
    # 同步 GroupNorm 的参数
    with torch.no_grad():
        block_bn.weight = nn.Parameter(batch_norm.weight.clone().unsqueeze(-1).broadcast_to(num_channels,
                                                                                         block_num).reshape(-1))
        block_bn.bias = nn.Parameter(batch_norm.bias.clone().unsqueeze(-1).broadcast_to(num_channels,
                                                                                     block_num).reshape(-1))

    # 创建一个示例输入张量
    input_tensor = torch.randn(batch_size, num_channels, height, width)

    
    # 显式分组进行 BatchNorm2d
    outputs = []
    for i in range(0, batch_size, block_size):
        chunk = input_tensor[i : i + block_size]
        outputs.append(batch_norm(chunk))
    explicit_bn_output = torch.cat(outputs, dim=0)

    # 基于 GroupNorm 的 BlockBatchNorm2d
    block_bn_output = block_bn(input_tensor)
    print(explicit_bn_output - block_bn_output)
    
    # 检查输出是否一致
    return torch.allclose(explicit_bn_output, block_bn_output, atol=1e-6)

def compare_alexnet():
    batch_size = 24
    num_channels = 3
    height = 256
    width = 256
    block_size = 3
    block_num = batch_size // block_size
    
    # 创建 AlexNet 实例
    alexnet = AlexNetV1()
    alexnet = alexnet.to('cuda:0')
    net_path = os.path.join(work_path, 'pretrained/siamfc/siamfc_alexnet_e50_download.pth')
    state_dict = torch.load(net_path, map_location=lambda storage, loc: storage)
    # 替换 state_dict 中的变量名
    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
    alexnet.load_state_dict(state_dict)
    
    # 创建 BlockAlexNet 实例
    block_alexnet = BlockAlexNetV1(alexnet, block_num, block_size)
    block_alexnet = block_alexnet.to('cuda:0')
    
    
    print("Successfully created AlexNet and BlockAlexNet instances.")

    # 创建一个示例输入张量
    img = cv2.imread('D:\MyFolders\project\CANN\SiamFC_CANN_v2\data\\train\GOT10K\\train\GOT-10k_Train_000014\\00000010.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB
    img = cv2.resize(img, (width, height))  # Resize image to match the desired height and width
    img = img.transpose((2, 0, 1))  # Transpose image dimensions from HWC to CHW
    input_tensor = torch.from_numpy(img).unsqueeze(0).broadcast_to(batch_size,
                                                                   num_channels,
                                                                   height,
                                                                   width).float().to('cuda:0')  # Convert image to tensor and move to GPU
    
    outputs = []
    for i in range(0, batch_size, block_size):
        chunk = input_tensor[i : i + block_size]    
        outputs.append(alexnet(chunk))
    alexnet_output = torch.cat(outputs, dim=0)
    block_alexnet_output = block_alexnet(input_tensor)

    print(torch.abs((alexnet_output - block_alexnet_output) / (alexnet_output + 1e-10) + 1e-10))
    print(f"Diff: {torch.max(torch.abs((alexnet_output - block_alexnet_output) / (alexnet_output + 1e-10)))}")
    print(f"Diff: {torch.mean(torch.abs((alexnet_output - block_alexnet_output) / alexnet_output))}")
    print(f"Equal: {torch.all(alexnet_output == block_alexnet_output)}")
    # assert torch.all(alexnet.conv2[0](alexnet_output) == block_alexnet.conv2[0](block_alexnet_output))
    
    
    import matplotlib.pyplot as plt

    # 计算相对误差
    relative_error = torch.abs((alexnet_output - block_alexnet_output) / (alexnet_output + 1e-10) + 1e-10)
    relative_error = torch.log10(relative_error)
    
    # 绘制直方图
    plt.hist(relative_error.flatten().detach().cpu().numpy(), bins=50)
    # plt.xscale('log')
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Relative Error')
    plt.show()
    
    return torch.allclose(alexnet_output, block_alexnet_output, atol=1e-6)

def compare_alexnet_3d():
    batch_size = 24
    num_channels = 3
    height = 256
    width = 256
    block_size = 3
    block_num = batch_size // block_size
    
    # 创建 AlexNet 实例
    alexnet = AlexNetV1()
    alexnet = alexnet.to('cuda:0')
    net_path = os.path.join(work_path, 'pretrained/siamfc/siamfc_alexnet_e50_download.pth')
    state_dict = torch.load(net_path, map_location=lambda storage, loc: storage)
    # 替换 state_dict 中的变量名
    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
    alexnet.load_state_dict(state_dict)
    
    # 创建 BlockAlexNet 实例
    block_alexnet = BlockAlexNetV1_3D(alexnet, block_num, block_size)
    block_alexnet = block_alexnet.to('cuda:0')
    
    
    print("Successfully created AlexNet and BlockAlexNet instances.")

    # 创建一个示例输入张量
    img = cv2.imread('D:\MyFolders\project\CANN\SiamFC_CANN_v2\data\\train\GOT10K\\train\GOT-10k_Train_000014\\00000010.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB
    img = cv2.resize(img, (width, height))  # Resize image to match the desired height and width
    img = img.transpose((2, 0, 1))  # Transpose image dimensions from HWC to CHW
    input_tensor = torch.from_numpy(img).unsqueeze(0).broadcast_to(batch_size,
                                                                   num_channels,
                                                                   height,
                                                                   width).float().to('cuda:0')  # Convert image to tensor and move to GPU
    
    outputs = []
    for i in range(0, batch_size, block_size):
        chunk = input_tensor[i : i + block_size]    
        outputs.append(alexnet(chunk))
    alexnet_output = torch.cat(outputs, dim=0)
    block_alexnet_output = block_alexnet(input_tensor)

    print(torch.abs((alexnet_output - block_alexnet_output) / (alexnet_output + 1e-10) + 1e-10))
    print(f"Diff: {torch.max(torch.abs((alexnet_output - block_alexnet_output) / (alexnet_output + 1e-10)))}")
    print(f"Diff: {torch.mean(torch.abs((alexnet_output - block_alexnet_output) / alexnet_output))}")
    print(f"Equal: {torch.all(alexnet_output == block_alexnet_output)}")
    # assert torch.all(alexnet.conv2[0](alexnet_output) == block_alexnet.conv2[0](block_alexnet_output))
    
    
    import matplotlib.pyplot as plt

    # 计算相对误差
    relative_error = torch.abs((alexnet_output - block_alexnet_output) / (alexnet_output + 1e-10) + 1e-10)
    relative_error = torch.log10(relative_error)
    
    # 绘制直方图
    plt.hist(relative_error.flatten().detach().cpu().numpy(), bins=50)
    # plt.xscale('log')
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Relative Error')
    plt.show()
    
    return torch.allclose(alexnet_output, block_alexnet_output, atol=1e-6)

def compare_alexnet_2d_3d():
    batch_size = 24
    num_channels = 3
    height = 256
    width = 256
    block_size = 3
    block_num = batch_size // block_size
    
    # 创建 AlexNet 实例
    alexnet = AlexNetV1()
    alexnet = alexnet.to('cuda:0')
    net_path = os.path.join(work_path, 'pretrained/siamfc/siamfc_alexnet_e50_download.pth')
    state_dict = torch.load(net_path, map_location=lambda storage, loc: storage)
    # 替换 state_dict 中的变量名
    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
    alexnet.load_state_dict(state_dict)
    
    # 创建 BlockAlexNet 实例
    block_alexnet_2d = BlockAlexNetV1(alexnet, block_num, block_size)
    block_alexnet_2d = block_alexnet_2d.to('cuda:0')
    block_alexnet_3d = BlockAlexNetV1_3D(alexnet, block_num, block_size)
    block_alexnet_3d = block_alexnet_3d.to('cuda:0')
    
    
    print("Successfully created AlexNet and BlockAlexNet instances.")

    # 创建一个示例输入张量
    img = cv2.imread('D:\MyFolders\project\CANN\SiamFC_CANN_v2\data\\train\GOT10K\\train\GOT-10k_Train_000014\\00000010.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB
    img = cv2.resize(img, (width, height))  # Resize image to match the desired height and width
    img = img.transpose((2, 0, 1))  # Transpose image dimensions from HWC to CHW
    input_tensor = torch.from_numpy(img).unsqueeze(0).broadcast_to(batch_size,
                                                                   num_channels,
                                                                   height,
                                                                   width).float().to('cuda:0')  # Convert image to tensor and move to GPU
    
    outputs = []
    for i in range(0, batch_size, block_size):
        chunk = input_tensor[i : i + block_size]    
        outputs.append(alexnet(chunk))
    alexnet_output = torch.cat(outputs, dim=0)
    block_alexnet_2d_output = block_alexnet_2d(input_tensor)
    block_alexnet_3d_output = block_alexnet_3d(input_tensor)
    

    print(torch.abs((block_alexnet_2d_output - block_alexnet_3d_output) / (block_alexnet_2d_output + 1e-10) + 1e-10))
    print(f"Diff: {torch.max(torch.abs((block_alexnet_2d_output - block_alexnet_3d_output) / (block_alexnet_2d_output + 1e-10)))}")
    print(f"Diff: {torch.mean(torch.abs((block_alexnet_2d_output - block_alexnet_3d_output) / block_alexnet_2d_output))}")
    print(f"Equal: {torch.all(block_alexnet_2d_output == block_alexnet_3d_output)}")
    
    
    import matplotlib.pyplot as plt

    # 计算相对误差
    relative_error = torch.abs((block_alexnet_2d_output - block_alexnet_3d_output) / (block_alexnet_2d_output + 1e-10) + 1e-10)
    relative_error = torch.log10(relative_error)
    
    # 绘制直方图
    plt.hist(relative_error.flatten().detach().cpu().numpy(), bins=50)
    # plt.xscale('log')
    plt.xlabel('Relative Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Relative Error')
    plt.show()
    
    return torch.allclose(block_alexnet_2d_output, block_alexnet_3d_output, atol=1e-6)


def simple_speed_test():
    batch_size = 24
    num_channels = 3
    height = 255
    width = 255
    block_size = 3
    block_num = batch_size // block_size
    

    # 同步 GroupNorm 的参数
    with torch.no_grad():
        block_bn.batch_norm.weight = nn.Parameter(batch_norm.weight.clone().unsqueeze(-1).broadcast_to(num_channels,
                                                                                         block_num).reshape(-1))
        block_bn.batch_norm.bias = nn.Parameter(batch_norm.bias.clone().unsqueeze(-1).broadcast_to(num_channels,
                                                                                     block_num).reshape(-1))

    # 创建一个示例输入张量
    input_tensor = torch.randn(batch_size, num_channels, height, width).to('cuda:0')
    
    # 进行 GPU 上的速度测试
    steps = 10000
    block_bn = block_bn.to('cuda:0')
    batch_norm = batch_norm.to('cuda:0')
    
    ## 先进行预热
    for _ in range(1000):
        bn_output = batch_norm(input_tensor)
        block_bn_output = block_bn(input_tensor)
    
    
    
    start_1 = torch.cuda.Event(enable_timing=True)
    end_1 = torch.cuda.Event(enable_timing=True)
    start_2 = torch.cuda.Event(enable_timing=True)
    end_2 = torch.cuda.Event(enable_timing=True)
    
    start_1.record()
    for _ in range(steps):
        bn_output = batch_norm(input_tensor)
    end_1.record()
    
    start_2.record()
    for _ in range(steps):
        block_bn_output = block_bn(input_tensor)
    end_2.record()  
    
    torch.cuda.synchronize()
    print(f'Time for Batchnorm2d: {start_1.elapsed_time(end_1) / steps} ms')
    print(f'Time for BlockBatchNorm2d: {start_2.elapsed_time(end_2) / steps} ms')

def alexnet_speed_test():
    batch_size = 24
    num_channels = 3
    height = 255
    width = 255
    block_size = 3
    block_num = batch_size // block_size
    
    # 创建 AlexNet 实例
    alexnet = AlexNetV1()
    # 创建 BlockAlexNet 实例
    block_alexnet = BlockAlexNetV1(alexnet, block_num, block_size)
    
    print("Successfully created AlexNet and BlockAlexNet instances.")
    
    
    # 创建一个示例输入张量
    input_tensor = torch.randn(batch_size, num_channels, height, width).to('cuda:0')
    
    # 进行 GPU 上的速度测试
    steps = 1000
    alexnet = alexnet.to('cuda:0')
    block_alexnet = block_alexnet.to('cuda:0')
    
    ## 先进行预热
    print("Start warming up...")
    for _ in tqdm(range(100)):
        bn_output = alexnet(input_tensor)
        block_bn_output = block_alexnet(input_tensor)
    print("Warm up finished.")
    
    
    start_1 = torch.cuda.Event(enable_timing=True)
    end_1 = torch.cuda.Event(enable_timing=True)
    start_2 = torch.cuda.Event(enable_timing=True)
    end_2 = torch.cuda.Event(enable_timing=True)
    
    print("Start timing AlexNet and BlockAlexNet...")
    start_1.record()
    for _ in tqdm(range(steps)):
        bn_output = alexnet(input_tensor)
    end_1.record()
    print("Finished timing AlexNet.")
    
    start_2.record()
    for _ in tqdm(range(steps)):
        block_bn_output = block_alexnet(input_tensor)
    end_2.record()  
    print("Finished timing BlockAlexNet.")
    
    
    torch.cuda.synchronize()
    print(f'Time for Batchnorm2d: {start_1.elapsed_time(end_1) / steps} ms')
    print(f'Time for BlockBatchNorm2d: {start_2.elapsed_time(end_2) / steps} ms')
    
    
    
    
# # 多次测试：模块
# all_tests_passed = all(compare_block_bn_and_explicit_bn() for _ in range(100))
# print("All tests passed:", all_tests_passed)


# 多次测试：神经网络
all_tests_passed = all(compare_alexnet_3d() for _ in range(100))
print("All tests passed:", all_tests_passed)

# 进行速度测试
# alexnet_speed_test()
