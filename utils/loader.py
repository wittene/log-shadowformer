import os

from dataset import DataLoaderTrain, DataLoaderVal
from options import LoadOptions

def get_training_data(base_dir, load_opts: LoadOptions):
    assert os.path.exists(base_dir)
    return DataLoaderTrain(base_dir, load_opts=load_opts)

def get_validation_data(base_dir, load_opts: LoadOptions, random_patch: int = None):
    assert os.path.exists(base_dir)
    return DataLoaderVal(base_dir, load_opts=load_opts, random_patch=random_patch)
