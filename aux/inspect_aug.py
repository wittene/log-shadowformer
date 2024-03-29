'''
Inspect image after data augmentation
'''

import numpy as np
import os
from torch.utils.data import DataLoader

import options
from utils import mkdir, get_training_data, save_img

BATCH_SIZE = 10

def get_batches(load_opts):
    '''
    Returns corresponding batches of target/shadow image(s) and filename(s) as a tuple
    Image is a torch.Tensor with shape (B, C, H, W)
    Discards rest of items
    '''
    loader = DataLoader(
        dataset = get_training_data(base_dir=opts.train_dir, load_opts=load_opts),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=1, drop_last=False
    )
    data = next(iter(loader))
    return ((data[0].cpu().detach().numpy().astype(np.float32), data[3]), (data[1].cpu().detach().numpy().astype(np.float32), data[4]))

def save_targets(batch, dir, prefix=''):
    '''
    Saves out effect of target adjustment for given batch
    Dir is a path to output folder
    Prefix is prepended to all filenames in batch
    '''
    mkdir(dir)
    imgs, fns = batch
    for i in range(len(fns)):
        img = imgs[i]
        fn = f'{prefix}-{fns[i]}' if prefix else fns[i]
        save_img((img*255.0).astype(np.ubyte), os.path.join(dir, fn))

if __name__ == '__main__':

    opts = options.TrainOptions(description='RGB denoising evaluation on validation set')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

    dir = os.path.join("results", "inspect_aug", opts.run_label)

    # Load training batch
    print('Loading...')
    target_batch, shadow_batch = get_batches(opts.load_opts)

    # Save out
    print('Saving images...')
    save_targets(target_batch, dir=dir, prefix='target')
    save_targets(shadow_batch, dir=dir, prefix='shadow')
    