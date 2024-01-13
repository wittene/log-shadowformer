import os

from dataset import DataLoaderTrain, DataLoaderVal
def get_training_data(rgb_dir, img_options, divisor, linear_transform, log_transform):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, divisor, linear_transform, log_transform, img_options, None)

def get_validation_data(rgb_dir, divisor, linear_transform, log_transform):
    print("GOT VAL DATA")
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, divisor, linear_transform, log_transform)
