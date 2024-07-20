import os
import subprocess

import cv2
import numpy as np
import torch
from . import gen_ops, img_ops, num_ops, visualization


class FailureSave:
    def __init__(self, save_dir, thresold=0.1):
        self.thresold = thresold
        self.save_dir = save_dir
    
    def init(self, seq_name):
        self.imgs = []
        self.rps = []
        self.rps_mmin = []
        self.rps_mmax = []
        self.imgs_rp = []
        self.annotations = []
        self.boxes = []
        self.endpoints = []
        self.failure_frame = []
        self.IoUs = []
        self.video_name = None
        self.lwindows = 20
        self.rwindows = 6
        self.save_info = {}
        self.seq_name = seq_name
        self.video_dir = os.path.join(self.save_dir, seq_name + ".mp4")
                
    
    def save(self):
        n = len(self.imgs)
        
        for i in range(n):
            if not self._check_IoU(i):
                continue
            lpoint, rpoint = max(0, i - self.lwindows + 1), min(n - 1, i + self.rwindows - 1)
            self.failure_frame.append(i - lpoint)
            self.endpoints.append((lpoint, rpoint))
        
        # 检查 img 的高是不是比 rp 还要小
        if self.imgs[0].shape[0] < self.rps[0].shape[0]:
            shape, to_h, to_w = self.imgs[0].shape, self.rps[0].shape[0], self.imgs[0].shape[1] + 10
            self.imgs = [self._padding(img, shape, to_h, to_w) for img in self.imgs]
            self.imgs_rp = [self._padding(img, shape, to_h, to_w) for img in self.imgs_rp]
        
        img_list = []
        anno_list = [] #(index, IoU, mmax, mmin)
        failure_frame_list = []
        
        for i, (lpoint, rpoint) in enumerate(self.endpoints):
            img = self.imgs[lpoint: rpoint + 1]
            img_rp = self.imgs_rp[lpoint: rpoint + 1]
            rp = self.rps[lpoint: rpoint + 1]
            index = ["#" + str(k + 1) for k in range(lpoint, rpoint + 1)] # 保存时跳过了第一帧
            IoU = self.IoUs[lpoint: rpoint + 1]
            rp_mmin = self.rps_mmin[lpoint: rpoint + 1]
            rp_mmax = self.rps_mmax[lpoint: rpoint + 1]
                        
            shape, to_h = rp[0].shape, img[0].shape[0]
            rp = [self._padding(r, shape, to_h, to_h) for r in rp]
            
            img, img_rp, rp = np.array(img), np.array(img_rp), np.array(rp)
            save_img = [np.concatenate([a, b, c], axis=1) for a, b, c, in zip(img, img_rp, rp)]
            save_anno = [(ind, iou, mmax, mmin) for ind, iou, mmax, mmin in zip(index, IoU, rp_mmax, rp_mmin)]
            
            failure_frame_list.append(len(img_list) + self.failure_frame[i])
            img_list.extend(save_img)
            anno_list.extend(save_anno)
            
        
        if len(img_list) == 0:
            return
        
        img_list = [self._add_text_to_image(img, index, IoU, mmax, mmin) 
                    for img, (index, IoU, mmax, mmin) in zip(img_list, anno_list)]
        failure_frame_list = [img_list[i] for i in failure_frame_list]
        
        # video saving
        if not os.path.exists(self.video_dir):
            frame_rate = 2
            height, width, _ = img_list[0].shape
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite existing file
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'bgr24',
                '-r', str(frame_rate),
                '-i', '-',  # Read input from pipe
                '-c:v', 'libx264',
                '-crf', '18',
                '-preset', 'fast',
                self.video_dir
            ]
            
            ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            
            for img in img_list:
                ffmpeg_process.stdin.write(img.tobytes())

            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
        
        # failure frame saving
        for i, img in enumerate(failure_frame_list):
            cv2.imwrite(os.path.join(self.save_dir, f"{self.seq_name}_{i}.jpg"), img)
            
    def calc_IoU(self):
        anno, box = self.annotations[-1], self.boxes[-1]
        anno, box= torch.tensor(anno, dtype=torch.float32), torch.tensor(box, dtype=torch.float32)
        anno[2], anno[3] = anno[0] + anno[2] - 1, anno[1] + anno[3] - 1
        box[2], box[3] = box[0] + box[2] - 1, box[1] + box[3] - 1
        anno, box = anno.unsqueeze(0), box.unsqueeze(0)
        IoU = box_iou(anno, box)
        IoU = IoU.item()
        self.IoUs.append(IoU)
    
    def _check_IoU(self, index):
        if index == 0:
            return False
        if self.IoUs[index] < self.thresold and \
           self.IoUs[index - 1] >= self.thresold:
            return True
        return False
    
    def _padding(self, img, from_shape, to_h, to_w):
        h_fs, w_fs = from_shape[0], from_shape[1]
        h_ts, w_ts = to_h, to_w
        top_pad = (h_ts - h_fs) // 2
        bottom_pad = h_ts - h_fs - top_pad
        left_pad = (w_ts - w_fs) // 2
        right_pad = w_ts - w_fs - left_pad
        
        if top_pad == 0 and bottom_pad == 0:
            return img
        
        new_img = np.zeros((h_ts, w_ts, 3), dtype=np.uint8)
        new_img[top_pad: -bottom_pad, left_pad: -right_pad, :] = img
        return new_img
    
    def _add_text_to_image(self, img, index, IoU, mmax, mmin):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 255, 255) # 黄色
        line_type = 2

        # 在左上角添加 frame
        img = self._add_text_with_background(img, index, (10, 30), font_scale, font_color, line_type)
        # 在左下角添加 IoU
        img = self._add_text_with_background(img, f"IoU: {IoU:.2f}", 
                                             (10, img.shape[0] - 10), font_scale, font_color, line_type)
        # 在右上角添加 mmax
        img = self._add_text_with_background(img, f"mmax: {mmax:.3f}", 
                                             (img.shape[1] - 250, 30), font_scale, font_color, line_type)
        # 在右下角添加 mmin
        img = self._add_text_with_background(img, f"mmin: {mmin:.3f}",
                                             (img.shape[1] - 250, img.shape[0] - 10), font_scale, font_color, line_type)
                                             
        return img
    
    def _add_text_with_background(self, img, text, position, font_scale, font_color, line_type):
        background_color = (0, 0, 0)
        alpha = 0.8
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_type = 2
        
        # 获取文字的宽度和高度
        text_size = cv2.getTextSize(text, font, font_scale, line_type)[0]
        
        # 计算背景矩形的坐标
        top_left = (position[0], position[1] - text_size[1] - 10)
        bottom_right = (position[0] + text_size[0] + 10, position[1] + 10)
        
        # 绘制带有透明度的背景矩形
        overlay = img.copy()
        cv2.rectangle(overlay, top_left, bottom_right, background_color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # 在背景上添加文字
        cv2.putText(img, text, (position[0], position[1]), font, font_scale, font_color, line_type)
        return img

class ComparisonSave:
    def __init__(self, save_dir):
        self.save_dir = save_dir
    
    def init(self, seq_name):
        self.imgs12 = []
        self.imgs1 = []
        self.imgs2 = [] 
        self.ress1 = []
        self.ress2 = []
        self.ress2_mmin = []
        self.ress2_mmax = []
        self.seq_name = seq_name
    
    def append(self, img12, img1, img2, res1, res2, res2_mmin, res2_mmax):
        self.imgs12.append(img12)
        self.imgs1.append(img1)
        self.imgs2.append(img2)
        self.ress1.append(res1)
        self.ress2.append(res2)
        self.ress2_mmin.append(res2_mmin)
        self.ress2_mmax.append(res2_mmax)
    
    def save(self, is_visualize=False):
        '''
            Parameters:
                self.imgs12: 融合了 1 算法和 2 算法得到的标注框的图片; shape=(h, w, 3)
                self.imgs1: 仅使用 1 算法得到的标注框的图片, 但融合了响应图; shape=(h, w, 3)
                self.imgs2: 仅使用 2 算法得到的标注框的图片, 但融合了响应图; shape=(h, w, 3)
                self.ress1: 算法 1 的响应图; shape=(k, k, 3)
                self.ress2: 算法 2 的响应图; shape=(k, k, 3)
                self.ress2_mmin, self.ress2_mmax: 算法 2 的响应图的 mmin 和 mmax; shape=(k, k, 3)
            Functions:
                将图片进行融合, 并保存为视频。具体地说，我们会把 imgs12 放大成 (2h, 2w, 3) 的图片，
                然后在其右侧添加 imgs1 和 imgs2, 然后继续在其右侧添加缩放为 (n, n)  
                最后整张图片的大小: (2h, 2w + w + h, 3)
            Default:
                一般而言, 算法 1 为 SiamFC, 算法 2 为对比算法(SiamFC+CANN)
        '''
        # 定义保存路径
        video_dir = os.path.join(self.save_dir, self.seq_name + ".mp4")
        mixed_imgs_dir = os.path.join(self.save_dir, self.seq_name)
        
        
        # 获取图像尺寸
        h, w, _ = self.imgs12[0].shape 
        k, _, _ = self.ress1[0].shape
        
        # 获取视频
        mixed_imgs = []
        for img12, img1, img2, res1, res2, res2_mmin, res2_mmax in zip(self.imgs12, self.imgs1, self.imgs2,
                                                                       self.ress1, self.ress2, 
                                                                       self.ress2_mmin, self.ress2_mmax):
            # 缩放响应图
            res1_resized = cv2.resize(res1, (h, h)) # (h, h, 3)
            res2_resized = cv2.resize(res2, (h, h))
            res2_resized_texted = visualization.add_minmax_to_image(res2_resized, res2_mmin, res2_mmax)
            
            # 放大 img12 到 (2n, 2m)
            img12_resized = cv2.resize(img12, (2 * w, 2 * h)) # (2h, 2w, 3)
            
            # 拼接 imgs1 和 imgs2
            img_combined = np.vstack((img1, img2)) # (2h, w, 3)
            
            # 拼接响应图
            res_combined = np.vstack((res1_resized, res2_resized_texted)) # (2h, h, 3)
            
            # 最终组合
            combined_img = np.hstack((img12_resized, img_combined)) # (2h, 2w + w, 3)
            combined_img = np.hstack((combined_img, res_combined)) # (2h, 2w + w + h, 3)
            mixed_imgs.append(combined_img)

            # 是否要在线可视化
            if is_visualize:
                cv2.imshow("mixed_imgs", combined_img)
                cv2.waitKey(0)
        
        # video saving
        if not os.path.exists(video_dir):
            frame_rate = 15
            height, width, _ = mixed_imgs[0].shape
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite existing file
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'bgr24',
                '-r', str(frame_rate),
                '-i', '-',  # Read input from pipe
                '-c:v', 'libx264',
                '-crf', '18',
                '-preset', 'fast',
                video_dir
            ]
            
            ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            if not os.path.exists(mixed_imgs_dir):
                os.makedirs(mixed_imgs_dir)
            
            for img in mixed_imgs:
                ffmpeg_process.stdin.write(img.tobytes())

            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
        
        # frame saving
        for i, img in enumerate(mixed_imgs):
            cv2.imwrite(os.path.join(mixed_imgs_dir, f"{self.seq_name}_{i}.jpg"), img)
            
class VideoSave:
    def __init__(self):
        self.imgs = []
    
    def append(self, img):
        self.imgs.append(img)
    def save(self, seq_names, videos_dir):
        result_video_path = os.path.join(work_path, os.path.join(videos_dir, seq_names + '.mp4'))
        if not os.path.isdir(videos_dir):
            os.makedirs(videos_dir)
        frame_rate = 15
        height, width, _ = self.imgs[0].shape
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite existing file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(frame_rate),
            '-i', '-',  # Read input from pipe
            '-c:v', 'libx264',
            '-crf', '18',
            '-preset', 'fast',
            result_video_path
        ]

        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        for img in self.imgs:
            ffmpeg_process.stdin.write(img.tobytes())

        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        
        pass