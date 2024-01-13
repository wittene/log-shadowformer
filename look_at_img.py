import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


def save_histogram(img_np:np.array, num_bins=50, torch_img=False, save_path="img_hist.png")->None:
    '''
    Given an image, generates and saves a histogram of the image pixel values.
    inputs:
        img: the image
        num_bins: (optional) number of bins for the histogram
        torch_img: (optional) set to true to reorder channels, rows, columns if the image is
                   in pytorch format
        save_path: (optional) the path where the image should be saved
    '''
    plt.hist(img_np.ravel(), bins=num_bins, density=True)
    plt.xlabel("pixel values")
    plt.ylabel("relative frequency")
    plt.title("distribution of pixels")
    plt.savefig(save_path)

img = cv2.imread("/work/SuperResolutionData/ShadowRemovalResults/ShadowFormer/pseudolog_no_layernorm0/ShadowFormer_istd/output_proj/100-1.exr", cv2.IMREAD_UNCHANGED)
print(np.min(img), np.max(img))

save_histogram(img)