## Part Segmentation on ShapeNet

### Data

We use the ShapeNetPart dataset (xyz, normals and labels) from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). Download the dataset and place it to `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`. 

### Usage

The settings are similar as in our classification experiment. To train a model for part segmentation (require 2 gpus for 2048 points input):

    python train.py --gpu_idx 0 1




