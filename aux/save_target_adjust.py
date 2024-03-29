'''
Saves the adjusted target image
'''

import numpy as np
import os
from torch.utils.data import DataLoader

import options
from utils import mkdir, get_validation_data, save_img, log_to_linear, apply_srgb

BATCH_SIZE = 1

def get_batch(load_opts):
    '''
    Returns batch of target(s) and target filename(s) as a tuple
    Target is a torch.Tensor with shape (B, C, H, W)
    Discards rest of items
    '''
    loader = DataLoader(
        dataset = get_validation_data(base_dir=opts.input_dir, load_opts=load_opts),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=1, drop_last=False
    )
    data = next(iter(loader))
    return (data[0].cpu().detach().numpy(), data[3])

def save_target_adjust(batch, dir, prefix=''):
    '''
    Saves out effect of target adjustment for given batch
    Dir is a path to output folder
    Prefix is prepended to all filenames in batch
    '''
    mkdir(dir)
    imgs, fns = batch
    for i in range(len(fns)):
        img = imgs[i]
        if opts.log_transform:
            img = log_to_linear(img, log_range=opts.log_range)
        if opts.linear_transform:
            img = apply_srgb(img, np.max(img))
        fn = f'{prefix}-{fns[i]}' if prefix else fns[i]
        save_img((img*255.0).astype(np.ubyte), os.path.join(dir, fn))

if __name__ == '__main__':

    opts = options.TestOptions(description='RGB denoising evaluation on validation set')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

    dir = os.path.join("results", "target_adjust", opts.run_label)

    # SRGB
    print('Outputting sRGB images...')
    save_target_adjust(
        get_batch(opts.update_load_opts(linear_transform=False, log_transform=False, target_adjust=False)),
        dir=dir,
        prefix='sRGB-orig'
    )
    save_target_adjust(
        get_batch(opts.update_load_opts(linear_transform=False, log_transform=False, target_adjust=True)),
        dir=dir,
        prefix='sRGB-adj'
    )
    # LINEAR
    print('Outputting linear images...')
    save_target_adjust(
        get_batch(opts.update_load_opts(linear_transform=True, log_transform=False, target_adjust=False)),
        dir=dir,
        prefix='linear-orig'
    )
    save_target_adjust(
        get_batch(opts.update_load_opts(linear_transform=True, log_transform=False, target_adjust=True)),
        dir=dir,
        prefix='linear-adj'
    )
    # LOG
    print('Outputting log images...')
    save_target_adjust(
        get_batch(opts.update_load_opts(linear_transform=True, log_transform=True, target_adjust=False)),
        dir=dir,
        prefix='log-orig'
    )
    save_target_adjust(
        get_batch(opts.update_load_opts(linear_transform=True, log_transform=True, target_adjust=True)),
        dir=dir,
        prefix='log-adj'
    )