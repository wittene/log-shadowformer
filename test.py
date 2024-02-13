
import numpy as np
import os
from tqdm import tqdm

import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from ptflops import get_model_complexity_info

import options
import utils
from utils import Checkpoint
from utils.loader import get_validation_data
from utils.pseudo_utils import *
from utils.image_utils import apply_srgb

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from sklearn.metrics import mean_squared_error as mse_loss

opts = options.TestOptions(description='RGB denoising evaluation on validation set')
load_opts = opts.load_opts
output_opts = opts.output_opts

MAX_VAL = 1 if not load_opts.log_transform else np.log(load_opts.log_range)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

utils.mkdir(output_opts.results_dir)
utils.mkdir(output_opts.residuals_dir)
residuals_eval_dir = os.path.join(output_opts.residuals_dir, "eval_best")
utils.mkdir(residuals_eval_dir)

test_dataset = get_validation_data(base_dir=opts.input_dir, load_opts=load_opts)
# test_dataset = get_validation_data(rgb_dir=opts.input_dir, load_opts=load_opts, random_patch=opts.tile)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

# Until tiling is debugged, random_patch is a workaround
# opts.tile = None

model_restoration = utils.get_arch(opts)
model_restoration = torch.nn.DataParallel(model_restoration)

checkpoint = utils.load_checkpoint(output_opts.weights_latest, map_location='cuda')
checkpoint.load_model(model_restoration)
print("===>Testing using weights: ", opts.weights)

model_restoration.cuda()
model_restoration.eval()

with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    rmse_val_rgb = []
    psnr_val_s = []
    ssim_val_s = []
    psnr_val_ns = []
    ssim_val_ns = []
    rmse_val_s = []
    rmse_val_ns = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = None
        rgb_gt = data_test[0].numpy().squeeze().transpose((1, 2, 0))
        rgb_noisy = data_test[1].cuda()
        mask = data_test[2].cuda()
        filenames = data_test[3]

        # Pad the input if not_multiple_of win_size * 8
        img_multiple_of = 8 * opts.win_size
        height, width = rgb_noisy.shape[2], rgb_noisy.shape[3]
        H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        rgb_noisy = F.pad(rgb_noisy, (0, padw, 0, padh), 'reflect')
        mask = F.pad(mask, (0, padw, 0, padh), 'reflect')

        # If tile is not set, evaluate on the full resolution
        # Otherwise, test the image tile by tile
        if opts.tile is None:
            restored, residual = model_restoration(rgb_noisy, mask)
        else:

            # TODO: debug this in log space to fix overlap issues

            # Set the tile size and overlap
            b, c, h, w = rgb_noisy.shape
            tile = min(opts.tile, h, w)
            assert tile % 8 == 0, "tile size should be multiple of 8"
            tile_overlap = opts.tile_overlap

            # Compute tile indices
            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            
            # Tensors used to compute final output (averaging the overlaps)
            out_patch_acc = torch.zeros(b, c, h, w).to(device=rgb_noisy.device, dtype=torch.float32)  # accumulates output patches
            out_patch_cts = torch.zeros_like(out_patch_acc)                 # counts number of patches at each position
            residual_patch_acc = torch.zeros_like(out_patch_acc)            # accumulates residual patches

            # Get tile outputs
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = rgb_noisy[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    mask_patch = mask[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    out_patch, residual_patch = model_restoration(in_patch, mask_patch)
                    out_patch_mask = torch.ones_like(out_patch)
                    # # if in log space, accumulate in linear to simplify averaging math
                    # if load_opts.log_transform:
                    #     out_patch = log_to_linear(out_patch, log_range=load_opts.log_range)
                    #     residual_patch = log_to_linear(residual_patch, log_range=load_opts.log_range)
                    out_patch_acc[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                    out_patch_cts[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
                    residual_patch_acc[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(residual_patch)
            
            # Create final output as an average
            restored = out_patch_acc.div_(out_patch_cts)
            residual = residual_patch_acc.div_(out_patch_cts)

            # # Convert back to log space if needed
            # if load_opts.log_transform:
            #     restored = linear_to_log(restored, log_range=load_opts.log_range)
            #     residual = linear_to_log(residual, log_range=load_opts.log_range)
        
        # Convert output into numpy for evaluation and unpad
        restored = torch.clamp(restored, 0, MAX_VAL).cpu().numpy().squeeze().transpose((1, 2, 0))
        restored = restored[:height, :width, :]
        residual = torch.clamp(residual, 0, MAX_VAL).cpu().numpy().squeeze().transpose((1, 2, 0))
        residual = residual[:height, :width, :]
        
        # E-Edit {
        if load_opts.log_transform:
            # model returns image in input space (log-space), convert output and target to linear for evaluation
            restored = log_to_linear(restored, log_range=load_opts.log_range)
            rgb_gt = log_to_linear(rgb_gt, log_range=load_opts.log_range)
            # mask = torch.multiply(mask, np.log(load_opts.log_range))
        # } E-Edit

        if opts.cal_metrics:
            bm = torch.where(mask == 0, torch.zeros_like(mask), torch.ones_like(mask))  #binarize mask
            bm = np.expand_dims(bm.cpu().numpy().squeeze(), axis=2)

            # calculate SSIM in gray space
            gray_restored = cv2.cvtColor(restored, cv2.COLOR_RGB2GRAY)
            gray_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2GRAY)
            ssim_val_rgb.append(ssim_loss(gray_restored, gray_gt, channel_axis=None))
            ssim_val_ns.append(ssim_loss(gray_restored * (1 - bm.squeeze()), gray_gt * (1 - bm.squeeze()), channel_axis=None))
            ssim_val_s.append(ssim_loss(gray_restored * bm.squeeze(), gray_gt * bm.squeeze(), channel_axis=None))

            psnr_val_rgb.append(psnr_loss(restored, rgb_gt))
            psnr_val_ns.append(psnr_loss(restored * (1 - bm), rgb_gt * (1 - bm)))
            psnr_val_s.append(psnr_loss(restored * bm, rgb_gt * bm))

            # calculate the RMSE in LAB space
            rmse_temp = np.abs(cv2.cvtColor(restored, cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2LAB)).mean() * 3
            rmse_val_rgb.append(rmse_temp)
            rmse_temp_s = np.abs(cv2.cvtColor(restored * bm, cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt * bm, cv2.COLOR_RGB2LAB)).sum() / bm.sum()
            rmse_temp_ns = np.abs(cv2.cvtColor(restored * (1-bm), cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt * (1-bm),
                                                                                                   cv2.COLOR_RGB2LAB)).sum() / (1-bm).sum()
            rmse_val_s.append(rmse_temp_s)
            rmse_val_ns.append(rmse_temp_ns)


        if opts.save_images:
            if load_opts.linear_transform:
                restored = apply_srgb(restored)
            utils.save_img((restored*255.0).astype(np.ubyte), os.path.join(output_opts.results_dir, filenames[0]))
        
        if opts.save_residuals:
            residual = np.clip(residual, 0, MAX_VAL)
            if load_opts.linear_transform:
                residual = utils.apply_srgb(residual, max_val=MAX_VAL)
            utils.save_img((residual*255.0).astype(np.ubyte), os.path.join(residuals_eval_dir, filenames[0]))

if opts.cal_metrics:
    psnr_val_rgb = sum(psnr_val_rgb)/len(test_dataset)
    ssim_val_rgb = sum(ssim_val_rgb)/len(test_dataset)
    psnr_val_s = sum(psnr_val_s)/len(test_dataset)
    ssim_val_s = sum(ssim_val_s)/len(test_dataset)
    psnr_val_ns = sum(psnr_val_ns)/len(test_dataset)
    ssim_val_ns = sum(ssim_val_ns)/len(test_dataset)
    rmse_val_rgb = sum(rmse_val_rgb) / len(test_dataset)
    rmse_val_s = sum(rmse_val_s) / len(test_dataset)
    rmse_val_ns = sum(rmse_val_ns) / len(test_dataset)
    print("PSNR: %f, SSIM: %f, RMSE: %f " %(psnr_val_rgb, ssim_val_rgb, rmse_val_rgb))
    print("SPSNR: %f, SSSIM: %f, SRMSE: %f " %(psnr_val_s, ssim_val_s, rmse_val_s))
    print("NSPSNR: %f, NSSSIM: %f, NSRMSE: %f " %(psnr_val_ns, ssim_val_ns, rmse_val_ns))

