import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import load_img, load_mask, load_val_mask, Augment_RGB_torch, adjust_target_colors
from options import LoadOptions
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, load_opts: LoadOptions = LoadOptions(), img_opts=None):
        super(DataLoaderTrain, self).__init__()
        
        # Get image filenames
        gt_dir = 'train_C'
        input_dir = 'train_A'
        mask_dir = 'train_B'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        mask_files = sorted(os.listdir(os.path.join(rgb_dir, mask_dir)))

        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files]
        self.mask_filenames = [os.path.join(rgb_dir, mask_dir, x) for x in mask_files]

        # Options for processing the images, e.g. patch size
        self.img_opts = img_opts

        # Number of targets, size of dataset
        self.tar_size = len(self.clean_filenames)

        # Load options
        self.load_opts = load_opts

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        # Load and adjust images

        clean = np.float32(load_img(self.clean_filenames[tar_index], load_opts=self.load_opts))
        noisy = np.float32(load_img(self.noisy_filenames[tar_index], load_opts=self.load_opts))
        mask = load_mask(self.mask_filenames[tar_index])
        
        if self.load_opts.target_adjust:
            clean = adjust_target_colors(clean, noisy, mask)
        
        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)
        mask = torch.from_numpy(np.float32(mask))

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        mask_filename = os.path.split(self.mask_filenames[tar_index])[-1]

        # Crop input and target

        ps = self.img_opts['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]
        mask = mask[r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        
        mask = getattr(augment, apply_trans)(mask)
        mask = torch.unsqueeze(mask, dim=0)
        
        return clean, noisy, mask, clean_filename, noisy_filename

##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, load_opts: LoadOptions = LoadOptions()):
        super(DataLoaderVal, self).__init__()

        # Get image filenames
        gt_dir = 'test_C'
        input_dir = 'test_A'
        mask_dir = 'test_B'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        mask_files = sorted(os.listdir(os.path.join(rgb_dir, mask_dir)))

        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files]
        self.mask_filenames = [os.path.join(rgb_dir, mask_dir, x) for x in mask_files]

        # Number of targets, size of dataset
        self.tar_size = len(self.clean_filenames)

        # Load options
        self.load_opts = load_opts

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        # Load and adjust images

        clean = np.float32(load_img(self.clean_filenames[tar_index], load_opts=self.load_opts))
        noisy = np.float32(load_img(self.noisy_filenames[tar_index], load_opts=self.load_opts))
        mask = load_val_mask(self.mask_filenames[tar_index])
        
        if self.load_opts.target_adjust:
            clean = adjust_target_colors(clean, noisy, mask)
        
        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)
        mask = torch.from_numpy(np.float32(mask))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        mask_filename = os.path.split(self.mask_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)
        mask = torch.unsqueeze(mask, dim=0)

        return clean, noisy, mask, clean_filename, noisy_filename
