'''
By Xiaolei Chen
Create in 2022/4/22 8:25
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius
import torch_geometric

import torch_geometric.transforms as T
from torch_cluster import knn_graph



def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # [B,1024,1024]每个点之间的距离
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)   因为前面算的距离是负的，所以是找距离最小的点作为邻居
    return idx


def get_knn_graph_feature(x, k=20, idx=None):  # DGCNN，用KNN动态建图
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')
    pass
    
    
    
    # idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    # idx = idx + idx_base    # 0～1023是batch1的点，1024～2047是batch2的点。。。。。

    # idx = idx.view(-1)
 
    # _, num_dims, _ = x.size()

    # x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    # feature = x.view(batch_size*num_points, -1)[idx, :]
    # feature = feature.view(batch_size, num_points, k, num_dims) # [B,1024,20,3]邻居点的特征
    # x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  #[B,1024,20,3]中心点的特征重复k次
    
    # feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()   # [B,6,1024,20] 每个邻居点特征为中心化特征concat上中心点特征
  
    # return feature

def get_knn_graph(x,k=5):
    batchsize = x.shape[0]
    numpoints = x.shape[1]

    batch = []
    for b in range(batchsize):
        for n in range(numpoints):
            batch.append(b)
    batch = torch.tensor(batch)
    x_reshape = x.reshape(batchsize*numpoints,-1)

    edge_index = knn_graph(x_reshape, k=k, batch=batch, loop=False)
    # """convert sparse edge index to dense adjacency matrices"""
    adj = torch_geometric.utils.to_dense_adj(edge_index,batch=batch)  # [B,1024,1024]
    
    return adj



def get_radius_graph(x,r=0.2):
    batchsize = x.shape[0]
    numpoints = x.shape[1]

    batch = []
    for b in range(batchsize):
        for n in range(numpoints):
            batch.append(b)
    batch = torch.tensor(batch).to(x.device)  # 创建tensor后移动到x所在的设备

    '''PyG PointNet2 example code, get a sparse edge index'''
    x_reshape = x.reshape(batchsize*numpoints,-1)
    row, col = radius(x_reshape, x_reshape, r, batch, batch,
                          max_num_neighbors=64)    
    
    edge_index = torch.stack([col, row], dim=0)

    """convert sparse edge index to dense adjacency matrices"""
    adj = torch_geometric.utils.to_dense_adj(edge_index,batch=batch)  # [B,1024,1024]
    
    return adj

def get_radius_sparse_graph(x,r=0.2):
    batchsize = x.shape[0]
    numpoints = x.shape[1]

    batch = []
    for b in range(batchsize):
        for n in range(numpoints):
            batch.append(b)
    batch = torch.tensor(batch)

    '''PyG PointNet2 example code, get a sparse edge index'''
    x_reshape = x.reshape(batchsize*numpoints,-1)
    row, col = radius(x_reshape, x_reshape, r, batch, batch,
                          max_num_neighbors=64)    
    
    edge_index = torch.stack([col, row], dim=0)

    # """convert sparse edge index to dense adjacency matrices"""
    # adj = torch_geometric.utils.to_dense_adj(edge_index,batch=batch)  # [B,1024,1024]
    
    return edge_index


if __name__ == "__main__":
    # pre_transform=T.KNNGraph(k=6)
    import torch
    from torch_cluster import knn_graph

    x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    batch = torch.tensor([0, 0, 0, 0])
    edge_index = knn_graph(x, k=2, batch=batch, loop=False)
    adj = torch_geometric.utils.to_dense_adj(edge_index,batch=batch)  # [B,1024,1024]

    print(edge_index)
    print(adj)

    y = torch.rand(2,5,3)
    adj = get_knn_graph(y,k=2)
    print(adj)