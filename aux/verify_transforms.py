'''
Evaluates how well each transform is reversed
'''

import numpy as np
import os
from torch.utils.data import DataLoader

import options
from utils import mkdir, get_validation_data, save_img, apply_srgb, log_to_linear

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
    return (data[0].cpu().detach().numpy().astype(np.float32), data[3])

def evaluate_transform(batch1, batch2):
    '''
    Given two batches, evaluate MSE
    batch1 taken as ideal
    '''
    imgs1, _ = batch1
    imgs2, _ = batch2
    return ((imgs1 - imgs2)**2).mean()

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

    opts = options.TestOptions(description='RGB denoising evaluation on validation set')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

    dir = os.path.join("results", "verify_xforms", opts.run_label)

    # Load batches of each condition
    print('Loading...')
    srgb_batch   = get_batch(opts.update_load_opts(linear_transform=False, log_transform=False, target_adjust=False))
    linear_batch = get_batch(opts.update_load_opts(linear_transform=True,  log_transform=False, target_adjust=False))
    log_batch    = get_batch(opts.update_load_opts(linear_transform=True,  log_transform=True,  target_adjust=False))

    print('\nEvaluating...')
    print(f'  -- sRGB Range:\t{np.min(srgb_batch[0])},\t{np.max(srgb_batch[0])}')
    print(f'  -- Linear Range:\t{np.min(linear_batch[0])},\t{np.max(linear_batch[0])}')
    print(f'  -- Log Range:\t\t{np.min(log_batch[0])},\t{np.max(log_batch[0])}')

    # Linear to sRGB
    print('linear-sRGB MSE:', end='\t')
    linear2srgb_batch = (apply_srgb(linear_batch[0]), linear_batch[1])
    print(evaluate_transform(srgb_batch, linear2srgb_batch))
    print(f'  -- Range:\t\t{np.min(linear2srgb_batch[0])},\t{np.max(linear2srgb_batch[0])}')

    # Log to linear
    print('log-linear MSE:', end='\t\t')
    log2linear_batch = (log_to_linear(log_batch[0], log_range=opts.log_range), log_batch[1])
    print(evaluate_transform(linear_batch, log2linear_batch))
    print(f'  -- Range:\t\t{np.min(log2linear_batch[0])},\t{np.max(log2linear_batch[0])}')

    # Log to sRGB
    print('log-sRGB MSE v1:', end='\t')
    log2sRGB_batch = (apply_srgb(log2linear_batch[0]), log2linear_batch[1])
    print(evaluate_transform(srgb_batch, log2sRGB_batch))
    print(f'  -- Range:\t\t{np.min(log2sRGB_batch[0])},\t{np.max(log2sRGB_batch[0])}')

    print('\nSaving images...')
    save_targets(srgb_batch, dir=dir, prefix='sRGB')
    save_targets(linear2srgb_batch, dir=dir, prefix='linear')
    save_targets(log2sRGB_batch, dir=dir, prefix='log')
