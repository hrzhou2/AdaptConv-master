import sys
from util import parameter_number
import os
import json
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
import time 
from torch.utils.data import Dataset, DataLoader, Sampler
from util import augmentation_transform

# Object number in dataset:
# catgory    | train | valid | test 
# ----------------------------------
# Airplane   |  1958 |   391 |   341
# Bag        |    54 |     8 |    14
# Cap        |    39 |     5 |    11
# Car        |   659 |    81 |   158
# Chair      |  2658 |   396 |   704
# Earphone   |    49 |     6 |    14
# Guitar     |   550 |    78 |   159
# Knife      |   277 |    35 |    80
# Lamp       |  1118 |   143 |   286
# Laptop     |   324 |    44 |    83
# Motorbike  |   125 |    26 |    51
# Mug        |   130 |    16 |    38
# Pistol     |   209 |    30 |    44
# Rocket     |    46 |     8 |    12
# Skateboard |   106 |    15 |    31
# Table      |  3835 |   588 |   848

PART_NUM = {
    "Airplane": 4,
    "Bag": 2,
    "Cap": 2,
    "Car": 4,
    "Chair": 4,
    "Earphone": 3,
    "Guitar": 3,
    "Knife": 2,
    "Lamp": 4,
    "Laptop": 2,
    "Motorbike": 6,
    "Mug": 2,
    "Pistol": 3,
    "Rocket": 3,
    "Skateboard": 3,
    "Table": 3,
}

CLASS_NUM = {
    "Airplane": 0,
    "Bag": 1,
    "Cap": 2,
    "Car": 3,
    "Chair": 4,
    "Earphone": 5,
    "Guitar": 6,
    "Knife": 7,
    "Lamp": 8,
    "Laptop": 9,
    "Motorbike": 10,
    "Mug": 11,
    "Pistol": 12,
    "Rocket": 13,
    "Skateboard": 14,
    "Table": 15,
}

TOTAL_PARTS_NUM = sum(PART_NUM.values())
TOTAL_CLASS_NUM = len(PART_NUM)

# For calculating mIoU
def get_valid_labels(category: str):
    assert category in PART_NUM
    base = 0
    for cat, num in PART_NUM.items():
        if category == cat:
            valid_labels = [base + i for i in range(num)]
            return valid_labels
        else: 
            base += num

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def normalize_pointcloud(pointcloud):
    center = pointcloud.mean(axis=0)
    pointcloud -= center
    distance = np.linalg.norm(pointcloud, axis=1)
    pointcloud /= distance.max()
    return pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ShapeNetDataset():
    def __init__(self, root, config, num_points=1024, split='train', normalize=True):
        self.num_points = num_points
        self.config = config
        self.split = split
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normalize = normalize
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k:v for k,v in self.cat.items()}
        #print(self.cat)
            
        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            #print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            #print(fns[0][0:-4])
            if split=='trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split=='train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split=='val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split=='test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..'%(split))
                exit(-1)
                
            #print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0]) 
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))
            
         
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        for cat in sorted(self.seg_classes.keys()):
            print(cat, self.seg_classes[cat])
        
        self.cache = {} # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
        
    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index]
            cat = self.datapath[index][0]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0] # cat name
            cls = self.classes[cat] # cat index
            cls = np.array([cls]).astype(np.int64)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:,0:3]
            if self.normalize:
                point_set = self.pc_normalize(point_set)
            normal = data[:,3:6]
            seg = data[:,-1].astype(np.int64)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, cls)
        
        if self.split == 'train' or self.split == 'trainval':
            point_set, normal = augmentation_transform(point_set, self.config, normals=normal)

        # sample to num_points
        if self.num_points <= point_set.shape[0]:
            sample_ids = np.random.permutation(point_set.shape[0])[:self.num_points]
        else:
            sample_ids = np.random.choice(point_set.shape[0], self.num_points, replace=True)
        #resample
        point_set = point_set[sample_ids, :]
        seg = seg[sample_ids]
        normal = normal[sample_ids, :]
        point_set = np.concatenate((point_set, normal), axis=1)

        mask = self.get_mask(cat)
        onehot = self.get_catgory_onehot(cls)
        obj_id = 0

        return cat, obj_id, point_set, seg, mask, onehot

    def pc_normalize(self, pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def get_mask(self, category):
        mask = torch.zeros(TOTAL_PARTS_NUM)
        mask[self.seg_classes[category]] = 1
        mask = mask.unsqueeze(0).repeat(self.num_points, 1)
        return mask

    def get_catgory_onehot(self, cat_id):
        onehot = torch.zeros(len(self.cat))
        onehot[cat_id] = 1
        return onehot

    def load_original_data(self, index):
        fn = self.datapath[index]
        cat = self.datapath[index][0] # cat name
        cls = self.classes[cat] # cat index
        cls = np.array([cls]).astype(np.int64)
        data = np.loadtxt(fn[1]).astype(np.float32)
        point_set = data[:,0:3]
        if self.normalize:
            point_set = self.pc_normalize(point_set)
        normal = data[:,3:6]
        seg = data[:,-1].astype(np.int64)
        point_set = np.concatenate((point_set, normal), axis=1)

        # onehot and mask
        onehot = self.get_catgory_onehot(cls)
        mask = torch.zeros(TOTAL_PARTS_NUM)
        mask[self.seg_classes[cat]] = 1
        mask = mask.unsqueeze(0).repeat(point_set.shape[0], 1)

        # repeat to 1024 at least
        ori_point_num = point_set.shape[0]
        if point_set.shape[0] < 640:
            cur_len = point_set.shape[0]
            res = np.array(point_set)
            while cur_len < 640:
                res = np.concatenate((res, point_set))
                cur_len += point_set.shape[0]
            point_set = res[:640, :]

        return cat, point_set, seg, onehot, mask, ori_point_num
        
    def __len__(self):
        return len(self.datapath)


class PartSegConfig():
    # Augmentations 
    augment_scale_anisotropic = True
    augment_symmetries = [False, False, False]
    normal_scale = True
    augment_shift = None
    augment_rotation = 'none'
    augment_scale_min = 0.8
    augment_scale_max = 1.25
    augment_noise = 0.002
    augment_noise_clip = 0.05
    augment_occlusion = 'none'

if __name__ == '__main__':
    data = ShapeNetDataset(root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', config=PartSegConfig(), split='trainval', num_points=1024)
    print('datapath', len(data.datapath), data.datapath[0])
    print('classes', data.classes)
    print('seg_classes', data.seg_classes)

    cat, obj_id, point_set, seg, mask, onehot = data[0]
    print(cat)
    print(point_set.shape, seg.shape, mask, onehot)
