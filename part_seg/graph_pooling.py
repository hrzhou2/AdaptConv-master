
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_nearest_index(target: "(bs, 3, v1)", source: "(bs, 3, v2)"):
    """
    Return: (bs, v1, 1)
    """
    inner = torch.bmm(target.transpose(1, 2), source) #(bs, v1, v2)
    s_norm_2 = torch.sum(source ** 2, dim=1) #(bs, v2)
    t_norm_2 = torch.sum(target ** 2, dim=1) #(bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
    nearest_index = torch.topk(d_norm_2, k= 1, dim= -1, largest= False)[1]
    return nearest_index

def indexing_neighbor(x: "(bs, dim, num_points0)", index: "(bs, num_points, k)" ):
    """
    Return: (bs, dim, num_points, neighbor_num)
    """
    batch_size, num_points, k = index.size()

    id_0 = torch.arange(batch_size).view(-1, 1, 1)

    x = x.transpose(2, 1).contiguous() # (bs, num_points, num_dims)
    feature = x[id_0, index] # (bs, num_points, k, num_dims)
    feature = feature.permute(0, 3, 1, 2).contiguous() # (bs, num_dims, num_points, k)
    '''
    idx_base = torch.arange(0, batch_size, device=index.device).view(-1, 1, 1)*num_points
    index = index + idx_base
    index = index.view(-1)

    x = x.transpose(2, 1).contiguous() # (bs, num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[index, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 

    feature = feature.permute(0, 3, 1, 2).contiguous() # (bs, num_dims, num_points, k)'''

    return feature


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    #x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = feature.permute(0, 3, 1, 2).contiguous()

    return feature, idx

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    points = points.transpose(2,1).contiguous()
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points.transpose(2,1).contiguous()

def index_feature(x, idx):
    """
    Input:
        x: input data, [bs, num_dims, num_points, k]
        idx: sample index data, [bs, new_npoints]
    Return:
        x:, indexed points data, [bs, num_dims, new_npoints, k]
    """
    x = x.permute(0, 2, 1, 3).contiguous() # (bs, num_points, num_dims, k)
    device = x.device
    B = x.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    x = x[batch_indices, idx, :]
    return x.permute(0, 2, 1, 3).contiguous()


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    xyz = xyz.transpose(2,1).contiguous()
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

#####################
# Graph Pooling Layer
#####################

class Pooling_fps(nn.Module):
    def __init__(self, pooling_rate, neighbor_num):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num

    def forward(self, 
                vertices: "(bs, 3, vertice_num)",
                feature_map: "(bs, channel_num, vertice_num)",
                idx):
        """
        Return:
            vertices_pool: (bs, 3, pool_vertice_num),
            feature_map_pool: (bs, channel_num, pool_vertice_num)
        """

        bs, _, vertice_num = vertices.size()
        neighbor_feature, _ = get_graph_feature(feature_map, k=self.neighbor_num, idx=idx) # (bs, num_dims, num_points, k)
        pooled_feature = torch.max(neighbor_feature, dim=-1)[0] #(bs, num_dims, num_points)

        new_npoints = int(vertice_num / self.pooling_rate)
        new_points_idx = farthest_point_sample(vertices, new_npoints) #(bs, new_npoints)
        vertices_pool = index_points(vertices, new_points_idx) # (bs, 3, new_npoints)
        feature_map_pool = index_points(pooled_feature, new_points_idx) #(bs, num_dims, new_npoints)

        return vertices_pool, feature_map_pool

