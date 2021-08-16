## Point Cloud Classification on ModelNet40

### Data

First, you may download the ModelNet40 dataset from [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip), and place it to `cls/data/modelnet40_ply_hdf5_2048`. We use the prepared data in HDF5 files for principle evaluation, where each object is already sampled to 2048 points. The experiments presented in the paper uses 1024 points for training and testing.

### Usage

To train a model for classification:

    python train.py 

Model and log files will be saved to `cls/models/train/` in default. After the training stage, you can test the model by:

    python train.py --eval 1

If you'd like to use your own data, you can modify `data.py` to change the data-loading path.



