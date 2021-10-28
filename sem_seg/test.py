
# Common libs
import signal
import os
import numpy as np
import sys
import torch
import torch.nn as nn
from os import makedirs, listdir
from os.path import exists, join
import time
import json
import argparse

# Dataset
from datasets.S3DIS import *
from torch.utils.data import DataLoader

from utils.config import Config
from models.architectures import Net

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KDTree


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='', help='Test name')
parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
parser.add_argument('--log', type=str, default='', help='Path to model.')
parser.add_argument('--model', type=str, default='', help='Chosen model.')
parser.add_argument('--dataset', type=str, default='./data/Stanford3dDataset_v1.2/', help='Path to S3DIS.')

args = parser.parse_args()


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def main():

    ###############################
    # Choose the model to visualize
    ###############################

    args.log = join('results', args.log)
    chosen_log = args.log

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = None

    # Choose to test on validation or test split
    on_val = True

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    ############################
    # Initialize the environment
    ############################

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if args.model == '':
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = args.model
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)
    print('Load pretrained: {}'.format(chosen_chkp))

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.
    #config.augment_symmetries = [False, False, False]
    #config.augment_rotation = 'none'
    #config.augment_scale_min = 0.99
    #config.augment_scale_max = 1.01
    #config.augment_noise = 0.0001
    #config.augment_color = 1.0

    #config.augment_noise = 0.0001
    #config.augment_symmetries = False
    #config.batch_num = 3
    #config.in_radius = 4
    config.validation_size = 200
    config.input_threads = 10
    

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    if on_val:
        set = 'validation'
    else:
        set = 'test'

    # Initiate dataset
    if config.dataset == 'S3DIS':
        test_dataset = S3DISDataset(config, args.dataset, set='validation', use_potentials=True)
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = S3DISCollate
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = Net(config, test_dataset.label_values, test_dataset.ignored_labels)

    # Define a visualizer class
    tester = ModelTester(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # Training
    tester.cloud_segmentation_test(net, test_loader, config)



# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#


class ModelTester:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, config, chkp_path=None, on_gpu=True):

        ############
        # Parameters
        ############

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device('cpu' if args.gpu_idx < 0 else 'cuda:{}'.format(args.gpu_idx))
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        # Test saving path
        self.record_file = None
        if config.saving:
            record_name = os.path.basename(chkp_path).split('.')[0]
            if args.name == '':
                self.test_path = join('test', config.saving_path.split('/')[-1], record_name)
            else:
                self.test_path = join('test', args.name, record_name)
                
            if not exists(self.test_path):
                makedirs(self.test_path)
            else:
                n = 0
                while True:
                    n += 1
                    new_log_dir = self.test_path + str(n)
                    if not exists(new_log_dir):
                        makedirs(new_log_dir)
                        self.test_path = new_log_dir
                        break
            print('Test path: {}'.format(self.test_path))

            if not exists(join(self.test_path, 'predictions')):
                makedirs(join(self.test_path, 'predictions'))
            
            self.record_file = open(join(self.test_path, 'TestInfo.txt'), 'w')
        else:
            self.test_path = None

        self.record(str(args))

        ##########################
        # Load previous checkpoint
        ##########################

        self.chkp_path = chkp_path
        checkpoint = torch.load(chkp_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint['epoch']
        net.eval()
        print("Model and training state restored.")

        return

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def record(self, info):
        print(info)
        if self.record_file:
            self.record_file.write(info + '\n')
            self.record_file.flush()

    def cloud_segmentation_test(self, net, test_loader, config, num_votes=100, debug=False):
        """
        Test method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        # Choose test smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        test_smooth = 0.95
        test_radius_ratio = 0.7
        softmax = torch.nn.Softmax(1)

        # Number of classes including ignored labels
        nc_tot = test_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        # Initiate global prediction over test clouds
        self.test_probs = [np.zeros((l.shape[0], nc_model)) for l in test_loader.dataset.input_labels]

        # If on validation directly compute score
        if test_loader.dataset.set == 'validation':
            val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in test_loader.dataset.label_values:
                if label_value not in test_loader.dataset.ignored_labels:
                    val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                 for labels in test_loader.dataset.validation_labels])
                    i += 1
        else:
            val_proportions = None

        #####################
        # Network predictions
        #####################

        test_epoch = 0
        last_min = -0.5

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start test loop
        while True:
            print('Initialize workers')
            for i, batch in enumerate(test_loader):

                # New time
                t = t[-1:]
                t += [time.time()]

                if i == 0:
                    print('Done in {:.1f}s'.format(t[1] - t[0]))

                if 'cuda' in self.device.type:
                    batch.to(self.device)

                # Forward pass
                with torch.no_grad():
                    outputs = net(batch, config)

                t += [time.time()]

                # Get probs and labels
                stacked_probs = softmax(outputs).cpu().detach().numpy()
                s_points = batch.points[0].cpu().numpy()
                lengths = batch.lengths[0].cpu().numpy()
                in_inds = batch.input_inds.cpu().numpy()
                cloud_inds = batch.cloud_inds.cpu().numpy()
                torch.cuda.synchronize(self.device)

                # Get predictions and labels per instance
                # ***************************************

                i0 = 0
                for b_i, length in enumerate(lengths):

                    # Get prediction
                    points = s_points[i0:i0 + length]
                    probs = stacked_probs[i0:i0 + length]
                    inds = in_inds[i0:i0 + length]
                    c_i = cloud_inds[b_i]

                    if 0 < test_radius_ratio < 1:
                        mask = np.sum(points ** 2, axis=1) < (test_radius_ratio * config.in_radius) ** 2
                        inds = inds[mask]
                        probs = probs[mask]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                    i0 += length

                # Average timing
                t += [time.time()]
                if i < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > 5.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f})'
                    print(message.format(test_epoch, i,
                                         100 * i / config.validation_size,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         1000 * (mean_dt[2])))

            # Update minimum od potentials
            new_min = torch.min(test_loader.dataset.min_potentials)
            print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))
            #print([np.mean(pots) for pots in test_loader.dataset.potentials])

            # Save predicted cloud
            if last_min + 1 < new_min:

                # Update last_min
                last_min += 1

                # Show vote results (On subcloud so it is not the good values here)
                if test_loader.dataset.set == 'validation':
                    self.record('\nConfusion on sub clouds: Test Epoch #{}'.format(test_epoch))
                    Confs = []
                    for i, file_path in enumerate(test_loader.dataset.files):

                        # Insert false columns for ignored labels
                        probs = np.array(self.test_probs[i], copy=True)
                        for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                            if label_value in test_loader.dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)

                        # Predicted labels
                        preds = test_loader.dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

                        # Targets
                        targets = test_loader.dataset.input_labels[i]

                        # Confs
                        Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                        if label_value in test_loader.dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    self.record(s + '\n')

                # Save real IoU once in a while
                if int(np.ceil(new_min)) % 10 == 0:

                    # Project predictions
                    self.record('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    proj_probs = []
                    for i, file_path in enumerate(test_loader.dataset.files):

                        print(i, file_path, test_loader.dataset.test_proj[i].shape, self.test_probs[i].shape)

                        print(test_loader.dataset.test_proj[i].dtype, np.max(test_loader.dataset.test_proj[i]))
                        print(test_loader.dataset.test_proj[i][:5])

                        # Reproject probs on the evaluations points
                        probs = self.test_probs[i][test_loader.dataset.test_proj[i], :]
                        proj_probs += [probs]

                    t2 = time.time()
                    self.record('Done in {:.1f} s\n'.format(t2 - t1))

                    # Show vote results
                    if test_loader.dataset.set == 'validation':
                        self.record('Confusion on full clouds')
                        t1 = time.time()
                        Confs = []
                        for i, file_path in enumerate(test_loader.dataset.files):

                            # Insert false columns for ignored labels
                            for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                                if label_value in test_loader.dataset.ignored_labels:
                                    proj_probs[i] = np.insert(proj_probs[i], l_ind, 0, axis=1)

                            # Get the predicted labels
                            preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                            # Confusion
                            targets = test_loader.dataset.validation_labels[i]
                            Confs += [fast_confusion(targets, preds, test_loader.dataset.label_values)]

                        t2 = time.time()
                        self.record('Done in {:.1f} s\n'.format(t2 - t1))

                        # Regroup confusions
                        C = np.sum(np.stack(Confs), axis=0)

                        # Remove ignored labels from confusions
                        for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
                            if label_value in test_loader.dataset.ignored_labels:
                                C = np.delete(C, l_ind, axis=0)
                                C = np.delete(C, l_ind, axis=1)

                        IoUs = IoU_from_confusions(C)
                        mIoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * mIoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        self.record('-' * len(s))
                        self.record(s)
                        self.record('-' * len(s) + '\n')

                    # Save predictions
                    print('Saving clouds')
                    t1 = time.time()
                    for i, file_path in enumerate(test_loader.dataset.files):

                        # Get file
                        points = test_loader.dataset.load_evaluation_points(file_path)

                        # Get the predicted labels
                        preds = test_loader.dataset.label_values[np.argmax(proj_probs[i], axis=1)].astype(np.int32)

                        # compute accuracy
                        targets = test_loader.dataset.validation_labels[i]
                        acc = accuracy_score(targets, preds)
                        macc = balanced_accuracy_score(targets, preds)
                        self.record('OA: {}, mAcc: {}'.format(acc, macc))

                        # Save plys
                        cloud_name = file_path.split('/')[-1]
                        test_name = join(self.test_path, 'predictions', cloud_name)
                        write_ply(test_name,
                                  [points, preds],
                                  ['x', 'y', 'z', 'preds'])

                        # Save ascii preds
                        if test_loader.dataset.set == 'test':
                            if test_loader.dataset.name.startswith('Semantic3D'):
                                ascii_name = join(self.test_path, 'predictions', test_loader.dataset.ascii_files[cloud_name])
                            else:
                                ascii_name = join(self.test_path, 'predictions', cloud_name[:-4] + '.txt')
                            np.savetxt(ascii_name, preds, fmt='%d')

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))

            test_epoch += 1

            # Break when reaching number of desired votes
            if last_min > num_votes:
                break

        return


if __name__ == '__main__':
    main()
