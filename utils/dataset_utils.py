import torch
import numpy as np
import cv2

### rotate and flip
class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor


### mix two images
class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy, gray_mask):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]
        gray_mask2 = gray_mask[indices]
        # gray_contour2 = gray_mask[indices]
        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2
        gray_mask = lam * gray_mask + (1-lam) * gray_mask2
        # gray_mask = torch.where(gray_mask>0.01, torch.ones_like(gray_mask), torch.zeros_like(gray_mask))
        # gray_contour = lam * gray_contour + (1-lam) * gray_contour2
        return rgb_gt, rgb_noisy, gray_mask


### Intensity/Color balance augmentation

class Color_Aug:
    def __init__(self, lower_bound=0.25, upper_bound=1.0) -> None:
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def intensity_aug(self, img: torch.Tensor):
        max_pixel = torch.max(img)
        coef = np.random.uniform(self.lower_bound / max_pixel, self.upper_bound / max_pixel)
        aug_img = torch.clamp(img * coef, 0, 1)
        return aug_img

    def color_aug(self, img: torch.Tensor):
        # Assumes tensor has shape (C, H, W)
        # Apply intensity aug separately to each channel
        aug_r = self.intensity_aug(img[0, :, :])
        aug_g = self.intensity_aug(img[1, :, :])
        aug_b = self.intensity_aug(img[2, :, :])
        aug_img = torch.stack([aug_r, aug_g, aug_b])
        return aug_img
    
    def aug(self, img, random_apply: bool = True):
        # apply each tansform randomly
        apply = lambda: np.random.randint(2) if random_apply else 1
        # do intensity, then color
        aug_img = self.intensity_aug(img) if apply() else img
        aug_img = self.color_aug(aug_img) if apply() else img
        return aug_img

### adjust shadow/no-shadow images

def dilate_mask(mask):
    kernel = np.ones((8,8), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    return dilation
    
def expand_to_three_channel(two_channel):
    three_channel = np.array([np.zeros_like(two_channel), 
                        np.zeros_like(two_channel), 
                        np.zeros_like(two_channel)])
    three_channel[0,:,:] = two_channel
    three_channel[1,:,:] = two_channel
    three_channel[2,:,:] = two_channel
    return three_channel

def rgb_mean(img, mask=None):
    '''Computes the mean value for each RGB channel separately'''
    r, g, b = np.split(img, 3, axis=2)
    if mask is not None:
        r = r[mask != 1]
        g = g[mask != 1]
        b = b[mask != 1]
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)
    return np.array([r_mean, g_mean, b_mean])

def adjust_target_colors(clean: np.array, noisy: np.array, mask: np.array):
    '''Images must be numpy arrays with shape (H, W, C)'''
    dilated_mask = dilate_mask(mask)
    clean_mean = rgb_mean(clean, mask=dilated_mask)
    noisy_mean = rgb_mean(noisy, mask=dilated_mask)
    gamma = noisy_mean / clean_mean
    clean_adjusted = np.clip(clean * gamma, 0, 1)
    return clean_adjusted