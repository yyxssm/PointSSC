from typing import List, Tuple, Union

import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import errno
import numpy as np
from tqdm import tqdm, trange
from typing import Optional


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
        
def fine_files():
    # 用于可视化, 找出3个模型都性能高的模型
    # 假定这是你的文件夹路径
    folder1 = '/home/csstuer/yyx/GS-Net-yyx/vis_save/Ada/gt'
    folder2 = '/home/csstuer/yyx/GS-Net-yyx/vis_save/PCN/gt'
    folder3 = '/home/csstuer/yyx/GS-Net-yyx/vis_save/SnowFlakeNet/gt'

    # 使用os.listdir获取每个文件夹中的文件名列表
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    files3 = set(os.listdir(folder3))

    # 使用集合的交集操作找出同时存在于三个文件夹中的文件名
    common_files = files1.intersection(files2).intersection(files3)
    return common_files
        
def read_bin_float32(path_file):
    """读取bin文件，读取类型为float32

    Args:
        path_file (str): 文件路径
    """
    return np.fromfile(path_file, dtype=np.float32)

def write_bin_float32(content, path_save):
    """输出bin文件，保存类型为float32

    Args:
        content (np.array): 待保存的文件，float32类型
        path_save (str): 文件保存路径
    """
    content.astype(np.float32).tofile(path_save)

def read_file_list(file_path):
    for _, _, file_names in os.walk(file_path):
        return file_names

class FocalLoss(nn.Module):
    def __init__(self, gamma: float, alpha: Optional[torch.Tensor] = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # [C,]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.gamma, self.alpha)


def focal_loss(pred_logit: torch.Tensor,
               label: torch.Tensor,
               gamma: float,
               alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
    # pred_logit [B, C]  or  [B, C, X1, X2, ...]
    # label [B, ]  or  [B, X1, X2, ...]
    B, C = pred_logit.shape[:2]  # batch size and number of categories
    if pred_logit.dim() > 2:
        # e.g. pred_logit.shape is [B, C, X1, X2]   
        pred_logit = pred_logit.reshape(B, C, -1)  # [B, C, X1, X2] => [B, C, X1*X2]
        pred_logit = pred_logit.transpose(1, 2)    # [B, C, X1*X2] => [B, X1*X2, C]
        pred_logit = pred_logit.reshape(-1, C)   # [B, X1*X2, C] => [B*X1*X2, C]   set N = B*X1*X2
    label = label.reshape(-1)  # [N, ]

    log_p = torch.log_softmax(pred_logit, dim=-1)  # [N, C]
    log_p = log_p.gather(1, label[:, None]).squeeze()  # [N,]
    p = torch.exp(log_p)  # [N,]
    
    if alpha is None:
        alpha = torch.ones((C,), dtype=torch.float, device=pred_logit.device)
    alpha = alpha.gather(0, label)  # [N,]
    
    loss = -1 * alpha * torch.pow(1 - p, gamma) * log_p
    return loss.sum() / alpha.sum()
