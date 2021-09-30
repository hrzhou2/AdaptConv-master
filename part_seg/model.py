import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import graph_pooling as gp


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim6=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim6 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 0:3], k=k)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature, idx


class AdaptiveConv(nn.Module):
    def __init__(self, k, in_channels, feat_channels, nhiddens, out_channels):
        super(AdaptiveConv, self).__init__()
        self.in_channels = in_channels
        self.nhiddens = nhiddens
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.k = k

        self.conv0 = nn.Conv2d(feat_channels, nhiddens, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(nhiddens, nhiddens*in_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(nhiddens)
        self.bn1 = nn.BatchNorm2d(nhiddens)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.residual_layer = nn.Sequential(nn.Conv2d(feat_channels, out_channels, kernel_size=1, bias=False),
                                            nn.BatchNorm2d(out_channels),
                                            )
        self.linear = nn.Sequential(nn.Conv2d(nhiddens, out_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(out_channels))

    def forward(self, points, feat, idx):
        # points: (bs, in_channels, num_points), feat: (bs, feat_channels/2, num_points)
        batch_size, _, num_points = points.size()

        x, _ = get_graph_feature(points, k=self.k, idx=idx) # (bs, in_channels, num_points, k)
        y, _ = get_graph_feature(feat, k=self.k, idx=idx) # (bs, feat_channels, num_points, k)

        kernel = self.conv0(y) # (bs, nhiddens, num_points, k)
        kernel = self.leaky_relu(self.bn0(kernel))
        kernel = self.conv1(kernel) # (bs, in*nhiddens, num_points, k)
        kernel = kernel.permute(0, 2, 3, 1).view(batch_size, num_points, self.k, self.nhiddens, self.in_channels) # (bs, num_points, k, nhiddens, in)

        x = x.permute(0, 2, 3, 1).unsqueeze(4) # (bs, num_points, k, in_channels, 1)
        x = torch.matmul(kernel, x).squeeze(4) # (bs, num_points, k, nhiddens)
        x = x.permute(0, 3, 1, 2).contiguous() # (bs, nhiddens, num_points, k)

        # nhiddens -> out_channels
        x = self.leaky_relu(self.bn1(x))
        x = self.linear(x) # (bs, out_channels, num_points, k)
        # residual: feat_channels -> out_channels
        residual = self.residual_layer(y)
        x += residual
        x = self.leaky_relu(x)

        x = x.max(dim=-1, keepdim=False)[0] # (bs, out_channels, num_points)

        return x

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, idx):
        # x: (bs, in_channels, num_points)
        x, _ = get_graph_feature(x, k=self.k, idx=idx) # (bs, in_channels, num_points, k)
        x = self.conv(x) # (bs, out_channels, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0] # (bs, out_channels, num_points)

        return x

class ConvLayer(nn.Module):
    def __init__(self, para, k, in_channels, feat_channels):
        super(ConvLayer, self).__init__()
        self.type = para[0]
        self.out_channels = para[1]
        self.k = k
        if self.type == 'adapt':
            self.layer = AdaptiveConv(k, in_channels, feat_channels, nhiddens=para[2], out_channels=para[1])
        elif self.type == 'graph':
            self.layer = GraphConv(feat_channels, self.out_channels, k)
        elif self.type == 'conv1d':
            self.layer = nn.Sequential(nn.Conv1d(int(feat_channels/2), self.out_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(self.out_channels),
                                    nn.LeakyReLU(negative_slope=0.2))
        else:
            raise ValueError('Unknown convolution layer: {}'.format(self.type))

    def forward(self, points, x, idx):
        # points: (bs, 3, num_points), x: (bs, feat_channels/2, num_points)
        if self.type == 'conv1d':
            x = self.layer(x)
            x = x.max(dim=-1, keepdim=False)[0] # (bs, num_dims)
        elif self.type == 'adapt':
            x = self.layer(points, x, idx)
        elif self.type == 'graph':
            x = self.layer(x, idx)

        return x


class Transform_Net(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super(Transform_Net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, out_channels*out_channels)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(out_channels, out_channels))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, self.out_channels, self.out_channels)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class Net(nn.Module):
    def __init__(self, args, class_num, cat_num, use_stn=True):
        super(Net, self).__init__()
        self.args = args
        self.k = args.k
        self.class_num = class_num
        self.cat_num = cat_num
        self.use_stn = use_stn

        # architecture
        self.in_channels = 6
        self.forward_para = [['adapt', 64, 64], 
                            ['adapt', 64, 64], 
                            ['pool', 4], 
                            ['adapt', 128, 64],
                            ['pool', 4], 
                            ['adapt', 256, 64], 
                            ['pool', 2], 
                            ['graph', 512],
                            ['conv1d', 1024]]
        self.agg_channels = 0

        # layers
        self.forward_layers = nn.ModuleList()
        feat_channels = 12
        for i, para in enumerate(self.forward_para):
            if para[0] == 'pool':
                self.forward_layers.append(gp.Pooling_fps(pooling_rate=para[1], neighbor_num=self.k))
            else:
                self.forward_layers.append(ConvLayer(para, self.k, self.in_channels, feat_channels))
                self.agg_channels += para[1]
                feat_channels = para[1]*2
        
        self.agg_channels += 64

        self.conv_onehot = nn.Sequential(nn.Conv1d(cat_num, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv1d = nn.Sequential(
            nn.Conv1d(self.agg_channels, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Conv1d(256, class_num, kernel_size=1),
            )

        if self.use_stn:
            self.stn = Transform_Net(in_channels=12, out_channels=3)


    def forward(self, x, onehot):
        # x: (bs, num_points, 6), onehot: (bs, cat_num)
        x = x.permute(0, 2, 1).contiguous() # (bs, 6, num_points)
        batch_size = x.size(0)
        num_points = x.size(2)

        if self.use_stn:
            x0, _ = get_graph_feature(x, k=self.k)
            t = self.stn(x0)
            p1 = torch.bmm(x[:,0:3,:].transpose(2, 1), t) # (bs, num_points, 3)
            p2 = torch.bmm(x[:,3:6,:].transpose(2, 1), t)
            x = torch.cat((p1, p2), dim=2).transpose(2, 1).contiguous() # (bs, 6, num_points)
        points = x[:,0:3,:] # (bs, 3, num_points)
        
        # forward
        feat_forward = []
        points_forward = [points]
        _, idx = get_graph_feature(points, k=self.k)
        for i, block in enumerate(self.forward_layers):
            if self.forward_para[i][0] == 'pool':
                points, x = block(points, x, idx)
                points_forward.append(points)
                _, idx = get_graph_feature(points, k=self.k)
            elif self.forward_para[i][0] == 'conv1d':
                x = block(points, x, idx)
                x = x.unsqueeze(2).repeat(1, 1, num_points)
                feat_forward.append(x)
            else:
                x = block(points, x, idx)
                feat_forward.append(x)

        # onehot
        onehot = onehot.unsqueeze(2)
        onehot_expand = self.conv_onehot(onehot)
        onehot_expand = onehot_expand.repeat(1, 1, num_points)

        # aggregating features from all layers
        x_agg = []
        points0 = points_forward.pop(0)
        points = None
        for i, para in enumerate(self.forward_para):
            if para[0] == 'pool':
                points = points_forward.pop(0)
            else:
                x = feat_forward.pop(0)
                if x.size(2) == points0.size(2):
                    x_agg.append(x)
                    continue
                idx = gp.get_nearest_index(points0, points)
                x_upsample = gp.indexing_neighbor(x, idx).squeeze(3)
                x_agg.append(x_upsample)
        x = torch.cat(x_agg, dim=1)
        x = torch.cat((x, onehot_expand), dim=1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1).contiguous() # (bs, num_points, class_num)

        return x


