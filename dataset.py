import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import load_imgs, Augment_RGB_torch
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

        # Load images
        clean, noisy, mask = load_imgs(
            clean_filename=self.clean_filenames[tar_index],
            noisy_filename=self.noisy_filenames[tar_index],
            mask_filename=self.mask_filenames[tar_index],
            load_opts=self.load_opts
        )

        clean = torch.from_numpy(np.float32(clean))
        noisy = torch.from_numpy(np.float32(noisy))
        mask  = torch.from_numpy(np.float32(mask))

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
    def __init__(self, rgb_dir, load_opts: LoadOptions = LoadOptions(), random_patch: int = None):
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
        self.random_patch = random_patch  # randomly crops image to a square patch of this size, if set

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        # Load images
        clean, noisy, mask = load_imgs(
            clean_filename=self.clean_filenames[tar_index],
            noisy_filename=self.noisy_filenames[tar_index],
            mask_filename=self.mask_filenames[tar_index],
            load_opts=self.load_opts
        )

        clean = torch.from_numpy(np.float32(clean))
        noisy = torch.from_numpy(np.float32(noisy))
        mask  = torch.from_numpy(np.float32(mask))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        mask_filename = os.path.split(self.mask_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)
        
        if self.random_patch:
            H = clean.shape[1]
            W = clean.shape[2]
            if H-self.random_patch==0:
                r=0
                c=0
            else:
                r = np.random.randint(0, H - self.random_patch)
                c = np.random.randint(0, W - self.random_patch)
            clean = clean[:, r:r + self.random_patch, c:c + self.random_patch]
            noisy = noisy[:, r:r + self.random_patch, c:c + self.random_patch]
            mask = mask[r:r + self.random_patch, c:c + self.random_patch]
        
        mask = torch.unsqueeze(mask, dim=0)

        return clean, noisy, mask, clean_filename, noisy_filename
