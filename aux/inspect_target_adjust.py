'''
Inspect how well the target adjustment is working.
'''

import os
import numpy as np
from matplotlib import pyplot as plt

from options import LoadOptions
from utils import mkdir, load_imgs, adjust_target_colors, dilate_mask, expand_to_three_channel, save_img

DATA_ROOT = "/work/SuperResolutionData/ShadowRemovalData/ISTD_Dataset/test/"
PNG = "101-3.png"

OUT_DIR = os.path.join("results", "inspect_target_adjust", PNG)
mkdir(OUT_DIR)

# Image paths
clean_filename = f'{DATA_ROOT}test_C/{PNG}'
noisy_filename = f'{DATA_ROOT}test_A/{PNG}'
mask_filename  = f'{DATA_ROOT}test_B/{PNG}'

# Load images without adjustment
print('Loading images...')
load_opts = LoadOptions()
clean, noisy, mask = load_imgs(clean_filename, noisy_filename, mask_filename, load_opts=load_opts)
dilated_mask2 = dilate_mask(mask)
dilated_mask3 = np.moveaxis(expand_to_three_channel(dilated_mask2), 0, 2)

print(f'Image shape: {clean.shape, noisy.shape}')
print(f'Mask shape: {mask.shape, dilated_mask3.shape}')
print(noisy[:,:,0].shape)

# Get adjusted target
clean_adjusted = adjust_target_colors(clean, noisy, mask)

# Save images for reference
save_img(clean*255, os.path.join(OUT_DIR, 'clean.png'))
save_img(clean_adjusted*255, os.path.join(OUT_DIR, 'clean_adjusted.png'))
save_img(noisy*255, os.path.join(OUT_DIR, 'noisy.png'))
save_img(dilated_mask3*255, os.path.join(OUT_DIR, 'mask_dilated.png'))
save_img(expand_to_three_channel(mask)*255, os.path.join(OUT_DIR, 'mask.png'))

# Get differences
diff_noadjust_r = noisy[:,:,0][dilated_mask2 != 1] - clean[:,:,0][dilated_mask2 != 1]
diff_noadjust_g = noisy[:,:,1][dilated_mask2 != 1] - clean[:,:,1][dilated_mask2 != 1]
diff_noadjust_b = noisy[:,:,2][dilated_mask2 != 1] - clean[:,:,2][dilated_mask2 != 1]
diff_noadjust = noisy[dilated_mask3 != 1] - clean[dilated_mask3 != 1]
diff_adjusted_r = noisy[:,:,0][dilated_mask2 != 1] - clean_adjusted[:,:,0][dilated_mask2 != 1]
diff_adjusted_g = noisy[:,:,1][dilated_mask2 != 1] - clean_adjusted[:,:,1][dilated_mask2 != 1]
diff_adjusted_b = noisy[:,:,2][dilated_mask2 != 1] - clean_adjusted[:,:,2][dilated_mask2 != 1]
diff_adjusted = noisy[dilated_mask3 != 1] - clean_adjusted[dilated_mask3 != 1]

# Plot difference histograms
print('Creating difference histograms...')
diff_hist_filename = os.path.join(OUT_DIR, 'diff_hist.png')
plt.hist(diff_noadjust, bins=10, alpha=0.5, label='Original')
plt.hist(diff_adjusted, bins=10, alpha=0.5, label='With adjustment')
plt.legend(loc='upper right')
plt.title("Differences in non-shadow regions")
plt.savefig(diff_hist_filename)
print(f'Difference histograms saved to {diff_hist_filename}')

# Compute SSE
print('Computing SSE...')
# ns_pixels = np.count_nonzero(dilated_mask3 != 1)
ns_pixels3 = dilated_mask3.shape[0] * dilated_mask3.shape[1] * dilated_mask3.shape[2] - np.squeeze(dilated_mask3[dilated_mask3!= 1].shape)
ns_pixels2 = dilated_mask2.shape[0] * dilated_mask2.shape[1] - np.squeeze(dilated_mask2[dilated_mask2!= 1].shape)
sse_noadjust_r = np.sum((diff_noadjust_r**2) / ns_pixels2)
sse_noadjust_g = np.sum((diff_noadjust_g**2) / ns_pixels2)
sse_noadjust_b = np.sum((diff_noadjust_b**2) / ns_pixels2)
sse_noadjust   = np.sum((diff_noadjust**2) / ns_pixels3)
sse_adjusted_r = np.sum((diff_adjusted_r**2) / ns_pixels2)
sse_adjusted_g = np.sum((diff_adjusted_g**2) / ns_pixels2)
sse_adjusted_b = np.sum((diff_adjusted_b**2) / ns_pixels2)
sse_adjusted   = np.sum((diff_adjusted**2) / ns_pixels3)

print(f'Number of pixels: {ns_pixels3}')
print(f'SSE (original):\t{sse_noadjust}')
print(f'  (R, G, B):\t{(sse_noadjust_r, sse_noadjust_g, sse_noadjust_b)}')
print(f'SSE (adjusted):\t{sse_adjusted}')
print(f'  (R, G, B):\t{(sse_adjusted_r, sse_adjusted_g, sse_adjusted_b)}')
