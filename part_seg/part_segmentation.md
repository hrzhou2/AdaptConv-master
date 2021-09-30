## Part Segmentation on ShapeNet

### Data

We use the ShapeNetPart dataset (xyz, normals and labels) from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip), which is prepared by [PointNet++](https://github.com/charlesq34/pointnet2). Download the dataset and place it to `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`. 

You can specific your own data directory by changing the `dataset` augment in `train.py`.

### Training

The settings are similar as in our classification experiment. To train a model on this dataset (require 2 gpus for 2048 points input):

    python train.py --gpu_idx 0 1

The models are saved every 10 epochs.

### Testing

To evaluate a model from `models/train/checkpoints/`:

    python test.py --log train --checkpoint epoch_099.pkl

Also, you can save the predicted obj files (ground truth, prediction, difference):

    python test.py --log train --checkpoint epoch_099.pkl --output ./results
