'''
Script to align target images to corresponding shadow image
Reference: https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/
'''

import os
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import cv2
from torch.utils.data import DataLoader

from utils import LoadOptions, mkdir, get_training_data, get_validation_data, save_img

ECC_TERMCRIT = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200,  1e-10)

def torch_to_np(img):
    np_img = img.cpu().detach().numpy().astype(np.float32).squeeze()
    if len(np_img.shape) == 3:
        np_img = np_img.transpose((1, 2, 0))
    np_img = (np_img * 255.).astype(np.ubyte)
    return np_img

if __name__ == '__main__':
    # example: python -m data.align --input_dir /work/SuperResolutionData/ShadowRemovalData/RawSR_Dataset/raw --dataset RawSR --img_type raw --motion_type affine

    # setup
    parser = ArgumentParser()
    parser.add_argument('--input_dir', required=True, type=str, help='Path to dataset')
    parser.add_argument('--gpu', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dataset', type=str, default='RawSR', help='dataset to use for eval: ISTD, RawSR')
    parser.add_argument('--img_type', type=str, default='raw', help='Input image type: sRGB, raw')
    parser.add_argument('--motion_type', type=str, default='translation', help='Motion type for ECC: translation, euclidean, affine, homography')
    
    opts = parser.parse_args()
    load_opts = LoadOptions(dataset=opts.dataset, img_type=opts.img_type, target_adjust=False)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

    
    # data
    datasets = {
        # 'train': get_training_data(opts.input_dir, load_opts=load_opts, img_opts={'patch_size': None, 'da': True}),
        'test': get_validation_data(opts.input_dir, load_opts=load_opts),
    }

    
    # ECC params
    if opts.motion_type == 'translation':
        warp_mode = cv2.MOTION_TRANSLATION
    elif opts.motion_type == 'euclidean':
        warp_mode = cv2.MOTION_EUCLIDEAN
    elif opts.motion_type == 'affine':
        warp_mode = cv2.MOTION_AFFINE
    elif opts.motion_type == 'homography':
        warp_mode = cv2.MOTION_HOMOGRAPHY
    else:
        warp_mode = cv2.MOTION_TRANSLATION

    # Do alignment
    for mode, dataset in datasets.items():
        print(f'Aligning {mode} images...')

        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
        output_dir = os.path.join(opts.input_dir, 'clean-affine')
        mkdir(output_dir)
        for data in tqdm(loader):

            target, shadow, mask, target_fn, shadow_fn = data
            target = torch_to_np(target)
            shadow = torch_to_np(shadow)
            mask   = torch_to_np(mask)

            # prepare inputs
            target_input = cv2.cvtColor(cv2.bitwise_and(target, target, mask=~mask), cv2.COLOR_RGB2GRAY)
            shadow_input = cv2.cvtColor(cv2.bitwise_and(shadow, shadow, mask=~mask), cv2.COLOR_RGB2GRAY)

            # Run the ECC algorithm. The results are stored in warp_matrix.
            warp_matrix = np.eye(3, 3, dtype=np.float32) if warp_mode == cv2.MOTION_HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)
            (cc, warp_matrix) = cv2.findTransformECC(shadow_input, target_input, warp_matrix, warp_mode, ECC_TERMCRIT)

            # Apply transform
            sz = target.shape
            if warp_mode == cv2.MOTION_HOMOGRAPHY :
                # Use warpPerspective for Homography
                target_aligned = cv2.warpPerspective(target, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            else:
                # Use warpAffine for Translation, Euclidean and Affine
                target_aligned = cv2.warpAffine(target, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            # Save
            # TODO: save out the warp matrices instead, to be used dynamically 
            save_img(target_aligned, os.path.join(output_dir, shadow_fn[0]))
            




    