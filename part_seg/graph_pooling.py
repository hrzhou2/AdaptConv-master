
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

# ***********************Pooling Layer***********************

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

class Pooling_strided(nn.Module):
    def __init__(self, pooling_rate, neighbor_num, in_channels):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.2))


    def forward(self, 
                vertices: "(bs, 3, vertice_num)",
                feature_map: "(bs, channel_num, vertice_num)",
                idx):
        """
        Return:
            vertices_pool: (bs, 3, pool_vertice_num),
            x: (bs, new_npoints, num_dims)
        """

        bs, _, vertice_num = vertices.size()
        neighbor_feature, _ = get_graph_feature(feature_map, k=self.neighbor_num, idx=idx) # (bs, num_dims, num_points, k)
        #neighbor_feature = neighbor_feature.permute(0,2,1,3) # (bs, num_points, num_dims, k)
        #pooled_feature = torch.max(neighbor_feature, dim=-1)[0] #(bs, num_dims, num_points)

        # downsample
        new_npoints = int(vertice_num / self.pooling_rate)
        new_points_idx = farthest_point_sample(vertices, new_npoints) #(bs, new_npoints)
        x = index_feature(neighbor_feature, new_points_idx) # (bs, num_dims, new_npoints, k)
        vertices_pool = index_points(vertices, new_points_idx) # (bs, 3, new_npoints)

        x = self.conv_layer(x)
        x = x.max(dim=-1, keepdim=False)[0] # (bs, num_dims, new_npoints)

        return vertices_pool, x

def test():
    import time
    bs = 8
    v = 1024
    dim = 6
    n = 20
    device='cuda:0'

    pool = Pool_layer(pooling_rate= 4, neighbor_num= 20).to(device)

    points = torch.randn(bs, 3, v).to(device)
    x = torch.randn(bs, dim, v).to(device)
    _, neighbor_idx = get_graph_feature(points, k=20)
    print('points: {}, x: {}, neighbor_idx: {}'.format(points.size(), x.size(), neighbor_idx.size()))

    points1, x1 = pool(points, x, neighbor_idx)
    print('points1: {}, x1: {}'.format(points1.size(), x1.size()))

    nearest_pool_1 = get_nearest_index(points, points1)
    print('nearest_pool_1: {}'.format(nearest_pool_1.size()))
    print(nearest_pool_1.device)

    x2 = indexing_neighbor(x1, nearest_pool_1).squeeze(3)
    print('x2: {}'.format(x2.size()))
    print(x2.device)



if __name__ == "__main__":
    test()
