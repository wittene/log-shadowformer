
import numpy as np
import os
import math
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
from utils.loader import get_validation_data
from utils.pseudo_utils import *
from utils.image_utils import apply_srgb

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from sklearn.metrics import mean_squared_error as mse_loss

args = options.TestOptions(description='RGB denoising evaluation on validation set')
MAX_LOG_VAL = math.log(args.log_range)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)
# utils.mkdir(args.output_proj_dir)

test_dataset = get_validation_data(args.input_dir, args.img_divisor, args.linear_transform, args.log_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

model_restoration = utils.get_arch(args)
model_restoration = torch.nn.DataParallel(model_restoration)

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()

img_multiple_of = 8 * args.win_size

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
        if args.log_transform:
            mask *= MAX_LOG_VAL
        filenames = data_test[3]

        # Pad the input if not_multiple_of win_size * 8
        height, width = rgb_noisy.shape[2], rgb_noisy.shape[3]
        H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        rgb_noisy = F.pad(rgb_noisy, (0, padw, 0, padh), 'reflect')
        mask = F.pad(mask, (0, padw, 0, padh), 'reflect')

        if args.tile is None:
            rgb_restored, _ = model_restoration(rgb_noisy, mask)
        else:
            # test the image tile by tile
            b, c, h, w = rgb_noisy.shape
            tile = min(args.tile, h, w)
            assert tile % 8 == 0, "tile size should be multiple of 8"
            tile_overlap = args.tile_overlap

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = torch.zeros(b, c, h, w).type_as(rgb_noisy)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = rgb_noisy[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    mask_patch = mask[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    out_patch, _ = model_restoration(in_patch, mask_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                    W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
            restored = E.div_(W)
        rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

        # Unpad the output
        rgb_restored = rgb_restored[:height, :width, :]
        # output_proj = output_proj[:height, :width, :]

        if args.cal_metrics:
            bm = torch.where(mask == 0, torch.zeros_like(mask), torch.ones_like(mask))  #binarize mask
            bm = np.expand_dims(bm.cpu().numpy().squeeze(), axis=2)

            # calculate SSIM in gray space
            gray_restored = cv2.cvtColor(rgb_restored, cv2.COLOR_RGB2GRAY)
            gray_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2GRAY)
            ssim_val_rgb.append(ssim_loss(gray_restored, gray_gt, channel_axis=None))
            ssim_val_ns.append(ssim_loss(gray_restored * (1 - bm.squeeze()), gray_gt * (1 - bm.squeeze()), channel_axis=None))
            ssim_val_s.append(ssim_loss(gray_restored * bm.squeeze(), gray_gt * bm.squeeze(), channel_axis=None))

            psnr_val_rgb.append(psnr_loss(rgb_restored, rgb_gt))
            psnr_val_ns.append(psnr_loss(rgb_restored * (1 - bm), rgb_gt * (1 - bm)))
            psnr_val_s.append(psnr_loss(rgb_restored * bm, rgb_gt * bm))

            # calculate the RMSE in LAB space
            rmse_temp = np.abs(cv2.cvtColor(rgb_restored, cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2LAB)).mean() * 3
            rmse_val_rgb.append(rmse_temp)
            rmse_temp_s = np.abs(cv2.cvtColor(rgb_restored * bm, cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt * bm, cv2.COLOR_RGB2LAB)).sum() / bm.sum()
            rmse_temp_ns = np.abs(cv2.cvtColor(rgb_restored * (1-bm), cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt * (1-bm),
                                                                                                   cv2.COLOR_RGB2LAB)).sum() / (1-bm).sum()
            rmse_val_s.append(rmse_temp_s)
            rmse_val_ns.append(rmse_temp_ns)


        if args.save_images:
            # output_proj[output_proj < 0] = 0
            # output_proj = output_proj / np.max(output_proj)
            if args.linear_transform or args.log_transform:
                rgb_restored = apply_srgb(rgb_restored)
                # output_proj = apply_srgb(output_proj)
            utils.save_img((rgb_restored*255.0).astype(np.ubyte), os.path.join(args.result_dir, filenames[0]))
            # utils.save_img((output_proj*255.0).astype(np.ubyte), os.path.join(args.output_proj_dir, filenames[0]))
            # cv2.imwrite(os.path.join(args.output_proj_dir, filenames[0][:-3] + "exr"), cv2.cvtColor(output_proj.astype(np.float32), cv2.COLOR_RGB2BGR))

if args.cal_metrics:
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

