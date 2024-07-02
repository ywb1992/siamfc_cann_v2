import torch
from torch.utils.data import DataLoader

def my_collate_fn(batch):
    '''
    Functions: 不进行张量化, 返回原始数据
    '''
    # 解包batch中的所有数据
    seq_imgs_path_list, seq_annos_list, seq_lens_list = zip(*batch)
    # seq_imgs_path: k*1, seq_annos: k*4, seq_lens: k
    return (seq_imgs_path_list, seq_annos_list, seq_lens_list)

