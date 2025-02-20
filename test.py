
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

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import structural_similarity as ssim_loss
from sklearn.metrics import mean_squared_error as mse_loss
def psnr_loss(pred, target, mask=None):
    if mask is not None:
        pred_masked = pred[mask]
        target_masked = target[mask]
        mse = np.mean((pred_masked - target_masked) ** 2)
    else:
        mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1 / mse)
def trimean(arr):
    return (np.quantile(arr, 0.25) + 2*np.quantile(arr, 0.5) + np.quantile(arr, 0.75)) / 4


opts = options.TestOptions(description='RGB denoising evaluation on validation set')
load_opts = opts.load_opts
output_opts = opts.output_opts

MAX_VAL = 1 if not load_opts.log_transform else np.log(load_opts.log_range)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

output_opts.results_dir = os.path.join(output_opts.results_dir, load_opts.dataset)
utils.mkdir(output_opts.results_dir)
utils.mkdir(output_opts.residuals_dir)
residuals_eval_dir = os.path.join(output_opts.residuals_dir, "eval_best", load_opts.dataset)
utils.mkdir(residuals_eval_dir)
residue_sub_dir = os.path.join(residuals_eval_dir, "residue")
utils.mkdir(residue_sub_dir)
output_opts.diffs_dir = os.path.join(output_opts.diffs_dir, load_opts.dataset)
utils.mkdir(output_opts.diffs_dir)

test_dataset = get_validation_data(base_dir=opts.input_dir, load_opts=load_opts)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

# Until tiling is debugged, random_patch is a workaround
# opts.tile = None

model_restoration = utils.get_arch(opts)
model_restoration = torch.nn.DataParallel(model_restoration)

checkpoint = utils.load_checkpoint(output_opts.weights_best, map_location='cuda')
checkpoint.load_model(model_restoration)
print(f"Testing [{output_opts.run_label}] on dataset [{load_opts.dataset}]")
print("===>Using weights: ", output_opts.weights_best)
print("===>With training epochs: ", checkpoint.epoch)

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
        filenames = data_test[4]

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
            restored, *residuals = model_restoration(rgb_noisy, mask)
            res1 = residuals[0]
            res2 = residuals[1] if len(residuals) == 2 else None
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
            out_patch_cts = torch.zeros_like(out_patch_acc)             # counts number of patches at each position
            res1_patch_acc = torch.zeros_like(out_patch_acc)            # accumulates residual patches
            res2_patch_acc = torch.zeros_like(out_patch_acc)            # accumulates residual patches

            # Get tile outputs
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = rgb_noisy[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    mask_patch = mask[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    out_patch, *residual_patches = model_restoration(in_patch, mask_patch)
                    res1_patch = residual_patches[0]
                    res2_patch = residual_patches[1] if len(residual_patches) == 2 else None
                    out_patch_mask = torch.ones_like(out_patch)
                    # # if in log space, accumulate in linear to simplify averaging math
                    # if load_opts.log_transform:
                    #     out_patch = log_to_linear(out_patch, log_range=load_opts.log_range)
                    #     residual_patch = log_to_linear(residual_patch, log_range=load_opts.log_range)
                    out_patch_acc[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                    out_patch_cts[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
                    res1_patch_acc[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(res1_patch)
                    if res2_patch is not None:
                        res2_patch_acc[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(res2_patch)
            
            # Create final output as an average
            restored = out_patch_acc.div_(out_patch_cts)
            res1 = res1_patch_acc.div_(out_patch_cts)
            res2 = res2_patch_acc.div_(out_patch_cts) if opts.split_residual else None

            # # Convert back to log space if needed
            # if load_opts.log_transform:
            #     restored = linear_to_log(restored, log_range=load_opts.log_range)
            #     residual = linear_to_log(residual, log_range=load_opts.log_range)
        
        # Unpad mask if needed (keep as torch tensor)
        mask = mask[:, :, :height, :width]
        # Convert output into numpy for evaluation and unpad
        restored = torch.clamp(restored, 0, MAX_VAL).cpu().numpy().squeeze().transpose((1, 2, 0))
        restored = restored[:height, :width, :]
        res1 = torch.clamp(res1, 0, MAX_VAL).cpu().numpy().squeeze().transpose((1, 2, 0))
        res1 = res1[:height, :width, :]
        if res2 is not None:
            res2 = torch.clamp(res2, 0, MAX_VAL).cpu().numpy().squeeze().transpose((1, 2, 0))
            res2 = res2[:height, :width, :]
        
        # E-Edit {
        # model returns image in input space, convert output and target to sRGB for evaluation
        if load_opts.log_transform:
            restored = log_to_linear(restored, log_range=load_opts.log_range)
            rgb_gt = log_to_linear(rgb_gt, log_range=load_opts.log_range)
            # mask = torch.multiply(mask, np.log(load_opts.log_range))
        if load_opts.linear_transform or load_opts.log_transform:
            # by here, max_val should always be 1
            restored = utils.apply_srgb(restored, max_val=1)
            rgb_gt = utils.apply_srgb(rgb_gt, max_val=1)
        # } E-Edit

        if opts.cal_metrics:
            bm = torch.where(mask == 0, torch.zeros_like(mask), torch.ones_like(mask))  #binarize mask
            bm = np.expand_dims(bm.cpu().numpy().squeeze(), axis=2)
            bm3 = np.concatenate([bm, bm, bm], axis=2).astype(bool)

            # calculate PSNR in sRGB space
            psnr_val_rgb.append(psnr_loss(restored, rgb_gt, mask=None))
            psnr_val_ns.append(psnr_loss(restored, rgb_gt, mask=~bm3))
            psnr_val_s.append(psnr_loss(restored, rgb_gt, mask=bm3))

            # calculate SSIM in gray space
            gray_restored = cv2.cvtColor(restored, cv2.COLOR_RGB2GRAY)
            gray_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2GRAY)
            ssim_val_rgb.append(ssim_loss(gray_restored, gray_gt, channel_axis=None))
            ssim_val_ns.append(ssim_loss(gray_restored * (1 - bm.squeeze()), gray_gt * (1 - bm.squeeze()), channel_axis=None))
            ssim_val_s.append(ssim_loss(gray_restored * bm.squeeze(), gray_gt * bm.squeeze(), channel_axis=None))

            # calculate the RMSE in LAB space
            rmse_val_rgb.append(np.abs(cv2.cvtColor(restored, cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2LAB)).mean() * 3)
            rmse_val_s.append(np.abs(cv2.cvtColor(restored * bm, cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt * bm, cv2.COLOR_RGB2LAB)).sum() / bm.sum())
            rmse_val_ns.append(np.abs(cv2.cvtColor(restored * (1-bm), cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt * (1-bm), cv2.COLOR_RGB2LAB)).sum() / (1-bm).sum())


        if opts.save_images:
            utils.save_img((restored*255.0).astype(np.ubyte), os.path.join(output_opts.results_dir, filenames[0]))
        
        if opts.save_residuals:
            res1 = np.clip(res1, 0, MAX_VAL)
            utils.save_img((res1*255.0).astype(np.ubyte), os.path.join(residuals_eval_dir, filenames[0]))
        
        if opts.save_residuals and res2 is not None:
            res2 = np.clip(res2, 0, MAX_VAL)
            utils.save_img((res2*255.0).astype(np.ubyte), os.path.join(residue_sub_dir, filenames[0]))
        
        if opts.save_diffs:
            diff = np.absolute(rgb_gt - restored)
            diff = np.clip(diff, 0, 1)
            utils.save_img((diff*255.0).astype(np.ubyte), os.path.join(output_opts.diffs_dir, filenames[0]))

if opts.cal_metrics:

    psnr_val_rgb = np.array(psnr_val_rgb)
    ssim_val_rgb = np.array(ssim_val_rgb)
    psnr_val_s = np.array(psnr_val_s)
    ssim_val_s = np.array(ssim_val_s)
    psnr_val_ns = np.array(psnr_val_ns)
    ssim_val_ns = np.array(ssim_val_ns)
    rmse_val_rgb = np.array(rmse_val_rgb)
    rmse_val_s = np.array(rmse_val_s)
    rmse_val_ns = np.array(rmse_val_ns)

    # lowest 25% of values
    lo_25 = lambda arr: arr[arr <= np.percentile(arr, 25)]
    # highest 25% of values
    hi_25 = lambda arr: arr[arr >= np.percentile(arr, 75)]

    # mean
    print('Eval Mean (Overall):')
    print(f"PSNR: {np.mean(psnr_val_rgb)}, SSIM: {np.mean(ssim_val_rgb)}, RMSE: {np.mean(rmse_val_rgb)} ")
    print(f"SPSNR: {np.mean(psnr_val_s)}, SSSIM: {np.mean(ssim_val_s)}, SRMSE: {np.mean(rmse_val_s)} ")
    print(f"NSPSNR: {np.mean(psnr_val_ns)}, NSSSIM: {np.mean(ssim_val_ns)}, NSRMSE: {np.mean(rmse_val_ns)} ")
    print()

    print('Eval Mean (Best 25%):')
    print(f"PSNR: {np.mean(hi_25(psnr_val_rgb))}, SSIM: {np.mean(hi_25(ssim_val_rgb))}, RMSE: {np.mean(lo_25(rmse_val_rgb))} ")
    print(f"SPSNR: {np.mean(hi_25(psnr_val_s))}, SSSIM: {np.mean(hi_25(ssim_val_s))}, SRMSE: {np.mean(lo_25(rmse_val_s))} ")
    print(f"NSPSNR: {np.mean(hi_25(psnr_val_ns))}, NSSSIM: {np.mean(hi_25(ssim_val_ns))}, NSRMSE: {np.mean(lo_25(rmse_val_ns))} ")
    print()

    print('Eval Mean (Worst 25%):')
    print(f"PSNR: {np.mean(lo_25(psnr_val_rgb))}, SSIM: {np.mean(lo_25(ssim_val_rgb))}, RMSE: {np.mean(hi_25(rmse_val_rgb))} ")
    print(f"SPSNR: {np.mean(lo_25(psnr_val_s))}, SSSIM: {np.mean(lo_25(ssim_val_s))}, SRMSE: {np.mean(hi_25(rmse_val_s))} ")
    print(f"NSPSNR: {np.mean(lo_25(psnr_val_ns))}, NSSSIM: {np.mean(lo_25(ssim_val_ns))}, NSRMSE: {np.mean(hi_25(rmse_val_ns))} ")
    print()

    # trimean
    print('Eval Trimean:')
    print(f"PSNR: {trimean(psnr_val_rgb)}, SSIM: {trimean(ssim_val_rgb)}, RMSE: {trimean(rmse_val_rgb)} ")
    print(f"SPSNR: {trimean(psnr_val_s)}, SSSIM: {trimean(ssim_val_s)}, SRMSE: {trimean(rmse_val_s)} ")
    print(f"NSPSNR: {trimean(psnr_val_ns)}, NSSSIM: {trimean(ssim_val_ns)}, NSRMSE: {trimean(rmse_val_ns)} ")
    print()

    # median
    print('Eval Median:')
    print(f"PSNR: {np.median(psnr_val_rgb)}, SSIM: {np.median(ssim_val_rgb)}, RMSE: {np.median(rmse_val_rgb)} ")
    print(f"SPSNR: {np.median(psnr_val_s)}, SSSIM: {np.median(ssim_val_s)}, SRMSE: {np.median(rmse_val_s)} ")
    print(f"NSPSNR: {np.median(psnr_val_ns)}, NSSSIM: {np.median(ssim_val_ns)}, NSRMSE: {np.median(rmse_val_ns)} ")
    print()
