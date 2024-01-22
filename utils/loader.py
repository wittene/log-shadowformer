import os

from dataset import DataLoaderTrain, DataLoaderVal
from options import LoadOptions

def get_training_data(rgb_dir, load_opts: LoadOptions, img_opts):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, load_opts=load_opts, img_opts=img_opts)

def get_validation_data(rgb_dir, load_opts: LoadOptions):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, load_opts=load_opts)
