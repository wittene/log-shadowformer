'''
Inspect percent saturation of training and test sets.
'''

import os
import argparse

from tqdm import tqdm
import csv

import numpy as np
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader

from options import ProgramOptions, LoadOptions
from utils import get_training_data, get_validation_data


def is_saturated(img: np.array):
    '''
    A pixel is saturated if it has a 0 or 255 in any color channel.
    Returns a boolean mask, where True indicates a saturated pixel, plus a saturation percentage.
    '''
    rs = img[:, :, 0]
    gs = img[:, :, 1]
    bs = img[:, :, 2]
    sat_mask = (rs <= 0) | (rs >= 1) | (gs <= 0) | (gs >= 1) | (bs <= 0) | (bs >= 1)
    sat_pct = np.count_nonzero(sat_mask) / sat_mask.size
    return sat_mask, sat_pct


if __name__ == '__main__':
    '''
    Usage:
    python -m aux.inspect_saturation --base_dir /work/SuperResolutionData/ShadowRemovalData/ISTD_Dataset --dataset ISTD
    python -m aux.inspect_saturation --base_dir /work/SuperResolutionData/ShadowRemovalData/RawSR_Dataset_final/linear --dataset RawSR
    python -m aux.inspect_saturation --base_dir /work/SuperResolutionData/ShadowRemovalData/RawSR_Dataset_final/srgb --dataset RawSR-compressed
    '''

    # Set up args
    opts = LoadOptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, help='dir of dataset')
    ProgramOptions.__add_load_args__(parser)
    parser.parse_args(namespace=opts)

    # Output
    out_dir = os.path.join("results", "inspect_saturation")
    os.makedirs(out_dir, exist_ok=True)

    # Set up dataset(s)
    datasets = dict()
    if opts.dataset == 'ISTD':
        datasets[f'{opts.dataset}-train'] = get_training_data(base_dir=os.path.join(opts.base_dir, 'train'), load_opts=opts)
        datasets[f'{opts.dataset}-test'] = get_validation_data(base_dir=os.path.join(opts.base_dir, 'test'), load_opts=opts)
    else:
        datasets[opts.dataset] = get_validation_data(base_dir=opts.base_dir, load_opts=opts)
    
    # Iterate
    for dataset, data in datasets.items():

        clean_vals = dict()
        noisy_vals = dict()

        loader = DataLoader(dataset=data, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
        for ii, data_test in enumerate(tqdm(loader), 0):

            clean = data_test[0].numpy().squeeze().transpose((1, 2, 0))
            noisy = data_test[1].numpy().squeeze().transpose((1, 2, 0))
            filename = data_test[4][0]

            clean_sat_mask, clean_sat_pct = is_saturated(clean)
            noisy_sat_mask, noisy_sat_pct = is_saturated(noisy)
            
            clean_vals[filename] = clean_sat_pct
            noisy_vals[filename] = noisy_sat_pct

        outputs = {'clean': clean_vals, 'noisy': noisy_vals}
        for output, output_vals in outputs.items():
            sorted_filenames = sorted(output_vals.keys(), key=lambda k: output_vals[k])
            ranked_filenames = {sorted_filenames[rank]: (rank + 1) for rank in range(len(sorted_filenames))}
            all_metrics = [(ranked_filenames[fn], fn, output_vals[fn]) for fn in sorted_filenames]
            header = ['Rank', 'Filename', 'SaturationPercentage']
            csv_path = os.path.join(out_dir, f'{dataset}-{output}-saturation.csv')
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                writer.writerows(all_metrics)
            print(f'Ranked saturation percentage output to {csv_path}')

