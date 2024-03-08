'''
Script to generate and save an augmented test dataset.
'''

import os
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader

from options import TestOptions
from utils import get_validation_data, save_img, mkdir

if __name__ == '__main__':
    # example: python -m data.gen_augment --run_label intensity --input_dir /work/SuperResolutionData/ShadowRemovalData/ISTD_Dataset/test --dataset ISTD --img_type sRGB --intensity_aug
    # example: python -m data.gen_augment --run_label color_balance --input_dir /work/SuperResolutionData/ShadowRemovalData/ISTD_Dataset/test --dataset ISTD --img_type sRGB --color_balance_aug
    # example: python -m data.gen_augment --run_label all_aug --input_dir /work/SuperResolutionData/ShadowRemovalData/ISTD_Dataset/test --dataset ISTD --img_type sRGB --intensity_aug --color_balance_aug
    # example: python -m data.gen_augment --input_dir /work/SuperResolutionData/ShadowRemovalData/RawSR_Dataset/raw --dataset RawSR --img_type raw --motion_type affine

    # setup
    opts = TestOptions(description='Generate a test set based on given load options')

    # ensure we are just loading and saving in sRGB
    opts.update_load_opts(
        patch_size=None,
        linear_transform=None,
        log_transform=None,
    )
    load_opts = opts.load_opts

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

    
    # data
    dataset = get_validation_data(opts.input_dir, load_opts=load_opts)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

    data_dir = dataset.dataset_dir
    output_dir = f'{data_dir.base_dir}-{opts.run_label}'
    target_dir = os.path.join(output_dir, data_dir.gt_dir)
    mkdir(target_dir)
    shadow_dir = os.path.join(output_dir, data_dir.input_dir)
    mkdir(shadow_dir)

    for data in tqdm(loader):

        # load augmented images
        target, shadow, mask, target_fn, shadow_fn = data

        # save augmented images
        save_img((target.cpu().detach().numpy().astype(np.float32) * 255.).astype(np.ubyte), os.path.join(target_dir, target_fn[0]))
        save_img((shadow.cpu().detach().numpy().astype(np.float32) * 255.).astype(np.ubyte), os.path.join(shadow_dir, shadow_fn[0]))
    