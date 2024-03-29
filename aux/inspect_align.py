'''
Inspect images after alignment
Reference: https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/
'''

import os
from tqdm import tqdm

import numpy as np
import cv2
from torch.utils.data import DataLoader

import options
from utils import mkdir, get_validation_data, save_img, load_npy

BATCH_SIZE = 20
WARP_MODE = cv2.MOTION_AFFINE  # cv2.MOTION_TRANSLATION, cv2.MOTION_HOMOGRAPHY, cv2.MOTION_AFFINE
ECC_TERMCRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000,  1e-10)

def torch_to_np(img):
    np_img = img.cpu().detach().numpy().astype(np.float32).squeeze()
    if len(np_img.shape) == 3:
        np_img = np_img.transpose((1, 2, 0))
    np_img = (np_img * 255.).astype(np.ubyte)
    return np_img

def get_batches(load_opts):
    '''
    Returns batch as a tuple
    Image is a numpy array with shape (H, W, C) with range [0,255]
    Discards rest of items
    '''
    loader = DataLoader(
        dataset = get_validation_data(base_dir=opts.input_dir, load_opts=load_opts),
        batch_size=1, shuffle=False, num_workers=1, drop_last=False
    )
    for ii, data in enumerate(loader):
        if ii == BATCH_SIZE:
            break
        target, shadow, mask, target_fn, shadow_fn = data
        # convert images back to numpy/cv2 format
        target = torch_to_np(target)
        shadow = torch_to_np(shadow)
        mask   = torch_to_np(mask)
        yield target, shadow, mask, target_fn[0], shadow_fn[0]

if __name__ == '__main__':

    opts = options.TestOptions(description='RGB denoising evaluation on validation set')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

    opts.input_dir = '/work/SuperResolutionData/ShadowRemovalData/RawSR_Dataset/compressed'
    opts.update_load_opts(
        dataset='RawSR-compressed',
        img_type='srgb',
        target_adjust=True,
        motion_transform=None
    )

    dir = os.path.join("results", "inspect_align", opts.run_label)
    mkdir(dir)

    motion_transform_map = load_npy(os.path.join(opts.input_dir, 'affine.npy')).item()

    print('Aligning images...')
    for batch in tqdm(get_batches(opts.load_opts)):
        target, shadow, mask, target_fn, shadow_fn = batch

        warp_matrix = motion_transform_map[shadow_fn]
        
        # # Prepare images: remove shadow with mask, grayscale
        # target_input = cv2.cvtColor(cv2.bitwise_and(target, target, mask=~mask), cv2.COLOR_BGR2GRAY)
        # shadow_input = cv2.cvtColor(cv2.bitwise_and(shadow, shadow, mask=~mask), cv2.COLOR_BGR2GRAY)

        # # Init warp matrix based on warp mode
        # if WARP_MODE == cv2.MOTION_HOMOGRAPHY:
        #     warp_matrix = np.eye(3, 3, dtype=np.float32)
        # else:
        #     warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # # Run the ECC algorithm. The results are stored in warp_matrix.
        # (cc, warp_matrix) = cv2.findTransformECC(shadow_input, target_input, warp_matrix, WARP_MODE, ECC_TERMCRIT)

        # Do alignment
        sz = target.shape
        if WARP_MODE == cv2.MOTION_HOMOGRAPHY :
            # Use warpPerspective for Homography
            target_aligned = cv2.warpPerspective(target, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            target_aligned = cv2.warpAffine(target, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        # Save
        # save_img(target, os.path.join(dir, f'target-{shadow_fn}'))
        # save_img(shadow, os.path.join(dir, f'shadow-{shadow_fn}'))
        # save_img(target_aligned, os.path.join(dir, f'aligned-{shadow_fn}'))
        save_img(np.absolute(target.astype(np.float32)-shadow.astype(np.float32)).astype(np.ubyte), os.path.join(dir, f'diff_orig-{shadow_fn}'))
        save_img(np.absolute(target_aligned.astype(np.float32)-shadow.astype(np.float32)).astype(np.ubyte), os.path.join(dir, f'diff_aligned-{shadow_fn}'))


    '''
    # Read the images to be aligned
    im1 =  cv2.imread(&quot;images/image1.jpg&quot;);
    im2 =  cv2.imread(&quot;images/image2.jpg&quot;);
    
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    
    # Find size of image1
    sz = im1.shape
    
    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Specify the number of iterations.
    number_of_iterations = 5000;
    
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
    
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography
    im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
    # Use warpAffine for Translation, Euclidean and Affine
    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    
    # Show final results
    cv2.imshow(&quot;Image 1&quot;, im1)
    cv2.imshow(&quot;Image 2&quot;, im2)
    cv2.imshow(&quot;Aligned Image 2&quot;, im2_aligned)
    cv2.waitKey(0)
    '''
    