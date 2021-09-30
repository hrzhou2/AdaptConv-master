# Common libs
import signal
import os
import numpy as np
import sys
import torch
import torch.nn as nn
from os import makedirs, listdir
from os.path import exists, join, isdir
import time
import json
import argparse

# PLY reader
from utils.ply import read_ply, write_ply

from sklearn.neighbors import KDTree
import sklearn.metrics as metrics


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='', help='Test name')
parser.add_argument('--test', type=str, default='', help='Path to test file.')
parser.add_argument('--model', type=str, default='best_chkp', help='Path to model.')
parser.add_argument('--dataset', type=str, default='/home/data4/zhr/S3DIS/data/Stanford3dDataset_v1.2/', help='Path to S3DIS.')
parser.add_argument('--outdir', type=str, default='visualize', help='Path to test file.')

args = parser.parse_args()

label2class = {0: 'ceiling',
               1: 'floor',
               2: 'wall',
               3: 'beam',
               4: 'column',
               5: 'window',
               6: 'door',
               7: 'chair',
               8: 'table',
               9: 'bookcase',
               10: 'sofa',
               11: 'board',
               12: 'clutter'}

class2color = {'ceiling': [0,255,0],
               'floor':   [0,0,255],
               'wall':    [0,255,255],
               'beam':        [255,255,0],
               'column':      [255,0,255],
               'window':      [100,100,255],
               'door':        [200,200,100],
               'table':       [170,120,200],
               'chair':       [255,0,0],
               'sofa':        [200,100,100],
               'bookcase':    [10,200,100],
               'board':       [200,200,200],
               'clutter':     [50,50,50]}

class2label = {v: k for k, v in label2class.items()}
label2color = {l: class2color[c] for l, c in label2class.items()}
color_array = np.vstack([c for _, c in class2color.items()]).astype(np.uint8)


def load_cloud_data(indir, cloud_name):

    print('\nLoading cloud files: {}'.format(indir))
    t0 = time.time()

    # Get rooms of the current cloud
    cloud_folder = join(indir, cloud_name)
    room_folders = [join(cloud_folder, room) for room in listdir(cloud_folder) if isdir(join(cloud_folder, room))]

    # Initiate containers
    room_names = []
    room_points_list = []
    room_colors_list = []
    room_labels_list = []

    # Loop over rooms
    for i, room_folder in enumerate(room_folders):

        print('Cloud %s - Room %d/%d : %s' % (cloud_name, i+1, len(room_folders), room_folder.split('/')[-1]))

        room_points = np.empty((0, 3), dtype=np.float32)
        room_colors = np.empty((0, 3), dtype=np.uint8)
        room_labels = np.empty((0, 1), dtype=np.int32)

        for object_name in listdir(join(room_folder, 'Annotations')):

            if object_name[-4:] == '.txt':

                # Text file containing point of the object
                object_file = join(room_folder, 'Annotations', object_name)

                # Object class and ID
                tmp = object_name[:-4].split('_')[0]
                if tmp in class2label:
                    object_class = class2label[tmp]
                elif tmp in ['stairs']:
                    object_class = class2label['clutter']
                else:
                    raise ValueError('Unknown object name: ' + str(tmp))

                # Correct bug in S3DIS dataset
                '''
                if object_name == 'ceiling_1.txt':
                    with open(object_file, 'r') as f:
                        lines = f.readlines()
                    for l_i, line in enumerate(lines):
                        if '103.0\x100000' in line:
                            lines[l_i] = line.replace('103.0\x100000', '103.000000')
                    with open(object_file, 'w') as f:
                        f.writelines(lines)
                '''

                # Read object points and colors
                object_data = np.loadtxt(object_file, dtype=np.float32)

                # Stack all data
                room_points = np.vstack((room_points, object_data[:, 0:3].astype(np.float32)))
                room_colors = np.vstack((room_colors, object_data[:, 3:6].astype(np.uint8)))
                object_labels = np.full((object_data.shape[0], 1), object_class, dtype=np.int32)
                room_labels = np.vstack((room_labels, object_labels))
        
        room_names.append(room_folder.split('/')[-1])
        room_points_list.append(room_points)
        room_colors_list.append(room_colors)
        room_labels_list.append(room_labels)

    print('Done in {:.1f}s'.format(time.time() - t0))

    return room_names, room_points_list, room_colors_list, room_labels_list


def load_pred_data_ply(indir):

    print('\nLoading predicted ply files from: {}'.format(indir))
    t0 = time.time()

    cloud_names = []
    cloud_points = []
    cloud_preds = []

    # for each cloud
    for cloud_file in listdir(indir):

        if cloud_file[-4:] == '.ply':
            data = read_ply(join(indir, cloud_file))

            points = np.vstack((data['x'], data['y'], data['z'])).T
            preds = data['preds']

            cloud_names.append(cloud_file[:-4])
            cloud_points.append(points)
            cloud_preds.append(preds)

    print('Done in {:.1f}s'.format(time.time() - t0))

    return cloud_names, cloud_points, cloud_preds


def main():

    test_dir = join('test', args.test, args.model, 'predictions')
    print('Computing ply files from predicted results: {}'.format(test_dir))

    # output file
    outdir = join(args.outdir, args.test, args.model)
    if not exists(outdir):
        makedirs(outdir)

    cloud_names, cloud_points, cloud_preds = load_pred_data_ply(test_dir)

    for i, cloud_name in enumerate(cloud_names):
        print(cloud_points[i].shape, cloud_preds[i].shape)

        room_names, room_points, room_colors, room_labels = load_cloud_data(args.dataset, cloud_name)
        room_lengths = [points.shape[0] for points in room_points]

        cloud_dir = join(outdir, cloud_name)
        if not exists(cloud_dir):
            makedirs(cloud_dir)

        room_accuracy = []

        offset = 0
        for room_idx in range(len(room_names)):

            output_rgb = join(cloud_dir, room_names[room_idx] + '.ply')
            write_ply(output_rgb, 
                      [room_points[room_idx], room_colors[room_idx]],
                      ['x', 'y', 'z', 'red', 'green', 'blue'])

            output_gt = join(cloud_dir, room_names[room_idx] + '_gt.ply')
            gt_color = color_array[ np.squeeze(room_labels[room_idx]) ]
            write_ply(output_gt, 
                      [room_points[room_idx], gt_color],
                      ['x', 'y', 'z', 'red', 'green', 'blue'])
            
            gt_labels = np.squeeze(room_labels[room_idx])

            output_pred = join(cloud_dir, room_names[room_idx] + '_pred.ply')
            preds = cloud_preds[i][offset : offset+room_lengths[room_idx]]
            pred_color = color_array[preds]
            write_ply(output_pred, 
                      [room_points[room_idx], pred_color],
                      ['x', 'y', 'z', 'red', 'green', 'blue'])

            room_accuracy.append(metrics.accuracy_score(gt_labels, preds))
            offset += room_lengths[room_idx]

        print(room_accuracy)
        idx = np.argsort(room_accuracy)
        room_names = [room_names[idx_] for idx_ in idx]
        room_accuracy = [room_accuracy[idx_] for idx_ in idx]
        with open(join(cloud_dir, 'acc.txt'), 'w') as file:
            for name, acc in zip(room_names, room_accuracy):
                file.write('{:s} {:.4f}\n'.format(name, acc))


if __name__ == '__main__':
    main()
