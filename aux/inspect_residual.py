'''
Inspect a residual image
'''

import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import mkdir

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def save_histogram(img:np.array, num_bins=50, torch_img=False, save_path="img_hist.png")->None:
    '''
    Given an image, generates and saves a histogram of the image pixel values.
    inputs:
        img: the image
        num_bins: (optional) number of bins for the histogram
        torch_img: (optional) set to true to reorder channels, rows, columns if the image is
                   in pytorch format
        save_path: (optional) the path where the image should be saved
    '''
    plt.hist(img.ravel(), bins=num_bins, density=True)
    plt.xlabel("Pixel values")
    plt.ylabel("Relative frequency")
    plt.title("Distribution of pixels")
    plt.savefig(save_path)


if __name__ == '__main__':

    # Settings
    parser = argparse.ArgumentParser(description='Inspect a residual image')
    parser.add_argument('--run_label', type=str, required=True, help='Run label, used in output path')
    parser.add_argument('--img', type=str, required=True, help='Path to residual image')
    
    opts = parser.parse_args()
    dir = os.path.join("results", "inspect_residual", opts.run_label)
    mkdir(dir)

    # Read image
    print('Reading image...')
    img = cv2.imread(opts.img, cv2.IMREAD_UNCHANGED)
    print()

    # Output image info
    print('Saving histogram...')
    save_path = os.path.join(dir, os.path.basename(opts.img))
    save_histogram(img, save_path=save_path)
    print(f'Histogram saved to {save_path}')
    print()

    print(f'Image range: [{np.min(img)}, {np.max(img)}]')
    print()
