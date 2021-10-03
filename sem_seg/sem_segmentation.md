## S3DIS indoor segmentation

### Data

The S3DIS dataset can be downloaded <a href="https://goo.gl/forms/4SoGp4KtH1jfRqEj2">here (4.8 GB)</a>. 
Download the file named `Stanford3dDataset_v1.2.zip`, and move it to `data/Stanford3dDataset_v1.2`. You may also specific your own data directory by changing the `path` argument in `train.py`.

Compile the C++ extension modules for python located in `cpp_wrappers`. Open a terminal in this folder, and run:

    sh compile_wrappers.sh

* The code has been tested on one configuration:
    - PyTorch 1.8.1, CUDA 10.1


### Training

We train the network on a Tesla V100 gpu (to maintain the batch size). It will take a few more time in the first training. Simply run:

    python train.py

You may reduce the `batch_num` in `train.py` for some smaller 12GB gpus (`train_tiny.py`).

The models are saved in `results/train/checkpoints/` every 10 epochs.

### Testing

To test the model `current_chkp` in the previous run:

    python test.py --log ./results/train

And to test a model in epoch n:

    python test.py --log ./results/train --model epoch_0099.tar


### Acknowledgement

The S3DIS data processing was borrowed from <a href="https://github.com/HuguesTHOMAS/KPConv-PyTorch">KPConv</a>
