import os
import random

import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import load_imgs, Augment_RGB_torch, load_npy, Color_Aug, adjust_target_colors, srgb_to_rgb, linear_to_log, apply_srgb
from options import LoadOptions

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]

##################################################################################################
class DatasetTransforms():
    '''Helper for specifying transformations during load (images as np.array)'''

    # STATIC - transforms
    COLOR_AUG = Color_Aug(img_type=np.array)

    def pad_mask(noisy, mask):
        '''
        Pad mask to fit (if loading raw, rgb image may be bigger)
        '''
        if mask.shape != noisy.shape[:2]:
            # Calculate the padding
            h_diff = noisy.shape[0] - mask.shape[0]
            w_diff = noisy.shape[1] - mask.shape[1]
            pad_top = h_diff // 2
            pad_bottom = h_diff - pad_top
            pad_left = w_diff // 2
            pad_right = w_diff - pad_left
            padding = ((pad_top, pad_bottom), (pad_left, pad_right))
            # Apply padding
            mask = np.pad(mask, padding, 'reflect')
        return mask

    def resize(clean, noisy, mask, size):
        '''
        Resize images, maintain aspect ratio
        '''
        # get scaling factor using longest side
        scaling_factor = size / max(noisy.shape[0], noisy.shape[1])
        # apply
        clean = cv2.resize(clean, (int(clean.shape[1] * scaling_factor), int(clean.shape[0] * scaling_factor)), interpolation=cv2.INTER_AREA)
        noisy = cv2.resize(noisy, (int(noisy.shape[1] * scaling_factor), int(noisy.shape[0] * scaling_factor)), interpolation=cv2.INTER_AREA)
        mask  = cv2.resize(mask,  (int(mask.shape[1] * scaling_factor),  int(mask.shape[0] * scaling_factor)), interpolation=cv2.INTER_AREA)
        return clean, noisy, mask
    
    def motion_transform(clean, motion_matrix):
        '''
        Apply motion transform to target
        '''
        sz = clean.shape
        if motion_matrix.shape == (3,3):
            # Use warpPerspective for Homography
            return cv2.warpPerspective(clean, motion_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            return cv2.warpAffine(clean, motion_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    def color_augmentation(clean, noisy, intensity=True, color=False):
        '''
        Apply color augmentation in sRGB space
        '''
        if intensity and color:
            # apply transform 25/25/25/25
            [clean, noisy] = DatasetTransforms.COLOR_AUG.aug([clean, noisy])
        elif intensity:
            # apply transform 50/50
            if np.random.randint(2):
                [clean, noisy] = DatasetTransforms.COLOR_AUG.intensity_aug([clean, noisy])
        elif color:
            # apply transform 50/50
            if np.random.randint(2):
                [clean, noisy] = DatasetTransforms.COLOR_AUG.color_aug([clean, noisy])
        return clean, noisy
    
    def adjust_target(clean, noisy, mask):
        '''
        Adjust target colors
        '''
        return adjust_target_colors(clean, noisy, mask)

    def srgb_transform(clean, noisy):
        '''
        Convert linear to sRGB
        '''
        return apply_srgb(clean), apply_srgb(noisy)
    
    def linear_transform(clean, noisy):
        '''
        Convert sRGB to linear
        '''
        return srgb_to_rgb(clean), srgb_to_rgb(noisy)

    def log_transform(clean, noisy, log_range=65535):
        '''
        Convert linear to log
        '''
        return linear_to_log(clean, log_range=log_range), linear_to_log(noisy, log_range=log_range)


    # Construct
    
    def __init__(self, load_opts: LoadOptions = LoadOptions(), intensity_aug = None, color_aug = None, motion_matrix = None) -> None:
        self.load_opts = load_opts
        self.intensity_aug = intensity_aug
        self.color_aug = color_aug
        self.motion_matrix = motion_matrix
    
    def with_color_aug(self, intensity, color):
        '''
        Returns a copy with new color aug settings
        '''
        return DatasetTransforms(
            load_opts=self.load_opts,
            intensity_aug=intensity,
            color_aug=color,
            motion_matrix=self.motion_matrix
        )
    
    def with_motion(self, motion_matrix):
        '''
        Return a copy with set transformation matrix to use, changed with each image
        '''
        return DatasetTransforms(
            load_opts=self.load_opts,
            intensity_aug=self.intensity_aug,
            color_aug=self.color_aug,
            motion_matrix=motion_matrix
        )

    
    # Apply

    def __call__(self, clean: np.array, noisy: np.array, mask: np.array) -> tuple:
        '''
        Apply the transforms
        '''

        # geometric operations
        mask = DatasetTransforms.pad_mask(noisy, mask)
        if self.motion_matrix is not None:
            clean = DatasetTransforms.motion_transform(clean, 
                                                       self.motion_matrix)
        if self.load_opts.resize is not None:
            clean, noisy, mask = DatasetTransforms.resize(clean, noisy, mask, 
                                                          size=self.load_opts.resize)
        
        # color adjustments in linear space
        if self.load_opts.img_type == 'srgb':
            clean, noisy = DatasetTransforms.linear_transform(clean, noisy)
        if self.load_opts.target_adjust:
            clean = DatasetTransforms.adjust_target(clean, noisy, mask)
        if self.intensity_aug or self.color_aug:
            [clean, noisy] = DatasetTransforms.color_augmentation(clean, noisy, 
                                                                  intensity=self.intensity_aug, 
                                                                  color=self.color_aug)
        
        # linear/log transforms - already in linear space, so convert to sRGB or log
        if not self.load_opts.linear_transform and not self.load_opts.log_transform:
            # convert back to srgb
            clean, noisy = DatasetTransforms.srgb_transform(clean, noisy)
        elif self.load_opts.log_transform:
            # convert linear to log
            clean, noisy = DatasetTransforms.log_transform(clean, noisy, log_range=self.load_opts.log_range)
        
        return clean, noisy, mask


##################################################################################################
class DatasetDirectory():
    '''Helper for parsing dataset directory based on dataset'''

    VALID_DATASETS = {'ISTD', 'RawSR', 'RawSR-compressed'}
    VALID_MODES    = {'train', 'test'}

    def __init__(self, base_dir: str, dataset: str, mode: str) -> None:
        if dataset not in self.VALID_DATASETS:
            raise Exception(f'Invalid dataset: {dataset}. Dataset must be one of: {self.VALID_DATASETS}')
        if mode not in self.VALID_MODES:
            raise Exception(f'Invalid mode: {mode}. Mode must be one of: {self.VALID_MODES}')
        
        # Define dataset
        self.base_dir = base_dir
        self.dataset = dataset
        self.mode = mode
        
        # Locate sub-directories
        if dataset == 'ISTD':
            self.gt_dir = f'{self.mode}_C'
            self.input_dir = f'{self.mode}_A'
            self.mask_dir = f'{self.mode}_B'
        elif 'RawSR' in dataset:
            self.gt_dir = 'clean'
            self.input_dir = 'shadow'
            self.mask_dir = 'mask'
        
        # Sort and set filenames
        clean_files = sorted(os.listdir(os.path.join(base_dir, self.gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(base_dir, self.input_dir)))
        mask_files = sorted(os.listdir(os.path.join(base_dir, self.mask_dir)))
        if 'RawSR' in dataset:
            clean_files = [f"{x.split('-')[0]}-1.{x.split('.')[-1]}" for x in noisy_files]  # 1-N relation between noisy and clean files
        
        self.clean_filenames = [os.path.join(base_dir, self.gt_dir, x) for x in clean_files]
        self.noisy_filenames = [os.path.join(base_dir, self.input_dir, x) for x in noisy_files]
        self.mask_filenames = [os.path.join(base_dir, self.mask_dir, x) for x in mask_files]


##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, base_dir, load_opts: LoadOptions = LoadOptions()):
        super(DataLoaderTrain, self).__init__()

        # Load options
        self.load_opts = load_opts
        
        # Set up dataset
        self.dataset_dir = DatasetDirectory(base_dir=base_dir, dataset=self.load_opts.dataset, mode='train')

        # Number of targets, size of dataset
        self.tar_size = len(self.dataset_dir.clean_filenames)

        # Set up data transforms
        self.data_transforms: DatasetTransforms = DatasetTransforms(self.load_opts)
        if load_opts.motion_transform:
            self.motion_transform_map: dict = load_npy(os.path.join(self.dataset_dir.base_dir, f'{load_opts.motion_transform}.npy')).item()
        else:
            self.motion_transform_map = None

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        # Load images
        curr_data_transforms = self.data_transforms.with_color_aug(intensity=self.load_opts.intensity_aug, color=self.load_opts.color_balance_aug)
        if self.motion_transform_map is not None:
            curr_data_transforms = curr_data_transforms.with_motion(self.motion_transform_map[os.path.split(self.dataset_dir.noisy_filenames[tar_index])[-1]])
        clean, noisy, mask = load_imgs(
            clean_filename=self.dataset_dir.clean_filenames[tar_index],
            noisy_filename=self.dataset_dir.noisy_filenames[tar_index],
            mask_filename=self.dataset_dir.mask_filenames[tar_index],
            load_opts=self.load_opts,
            data_transforms=curr_data_transforms
        )

        clean = torch.from_numpy(np.float32(clean))
        noisy = torch.from_numpy(np.float32(noisy))
        mask  = torch.from_numpy(np.float32(mask))

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        clean_filename = os.path.split(self.dataset_dir.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.dataset_dir.noisy_filenames[tar_index])[-1]
        mask_filename = os.path.split(self.dataset_dir.mask_filenames[tar_index])[-1]

        # Crop input and target
        if self.load_opts.patch_size:
            ps = self.load_opts.patch_size
            H = clean.shape[1]
            W = clean.shape[2]
            r = np.random.randint(0, H - ps) if not H-ps else 0
            c = np.random.randint(0, W - ps) if not H-ps else 0
            clean = clean[:, r:r + ps, c:c + ps]
            noisy = noisy[:, r:r + ps, c:c + ps]
            mask = mask[r:r + ps, c:c + ps]

        # augmentation: geometric transformation
        apply_trans = transforms_aug[random.getrandbits(3)]
        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        
        mask = getattr(augment, apply_trans)(mask)

        mask = torch.unsqueeze(mask, dim=0)
        return clean, noisy, mask, clean_filename, noisy_filename

##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, base_dir, load_opts: LoadOptions = LoadOptions(), random_patch: int = None):
        super(DataLoaderVal, self).__init__()

        # Load options
        self.load_opts = load_opts
        self.random_patch = random_patch  # randomly crops image to a square patch of this size, if set

        # Set up dataset
        self.dataset_dir = DatasetDirectory(base_dir=base_dir, dataset=load_opts.dataset, mode='test')

        # Number of targets, size of dataset
        self.tar_size = len(self.dataset_dir.clean_filenames)

        # Set up data transforms
        self.data_transforms: DatasetTransforms = DatasetTransforms(self.load_opts)
        if load_opts.motion_transform:
            self.motion_transform_map: dict = load_npy(os.path.join(self.dataset_dir.base_dir, f'{load_opts.motion_transform}.npy')).item()
        else:
            self.motion_transform_map = None

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        # Load images
        curr_data_transforms = self.data_transforms.with_color_aug(intensity=self.load_opts.intensity_aug, color=self.load_opts.color_balance_aug)
        if self.motion_transform_map is not None:
            curr_data_transforms = curr_data_transforms.with_motion(self.motion_transform_map[os.path.split(self.dataset_dir.noisy_filenames[tar_index])[-1]])
        clean, noisy, mask = load_imgs(
            clean_filename=self.dataset_dir.clean_filenames[tar_index],
            noisy_filename=self.dataset_dir.noisy_filenames[tar_index],
            mask_filename=self.dataset_dir.mask_filenames[tar_index],
            load_opts=self.load_opts,
            data_transforms=self.data_transforms
        )

        clean = torch.from_numpy(np.float32(clean))
        noisy = torch.from_numpy(np.float32(noisy))
        mask  = torch.from_numpy(np.float32(mask))

        clean_filename = os.path.split(self.dataset_dir.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.dataset_dir.noisy_filenames[tar_index])[-1]
        mask_filename = os.path.split(self.dataset_dir.mask_filenames[tar_index])[-1]

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
