import os

from dataset import DataLoaderTrain, DataLoaderVal
from utils import LoadOpts

def get_training_data(rgb_dir, load_opts: LoadOpts, img_opts):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, load_opts=load_opts, img_opts=img_opts, target_transform=None)

def get_validation_data(rgb_dir, load_opts: LoadOpts):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, load_opts=load_opts)
