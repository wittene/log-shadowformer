import torch
import numpy as np
import os

from .image_utils import dilate_mask

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


### adjust shadow/no-shadow images
    
def expand_to_three_channel(two_channel):
    three_channel = np.array([np.zeros_like(two_channel), 
                        np.zeros_like(two_channel), 
                        np.zeros_like(two_channel)])
    three_channel[0,:,:] = two_channel
    three_channel[1,:,:] = two_channel
    three_channel[2,:,:] = two_channel
    return three_channel

def adjust_target_colors(noshadowimg, shadowimg, mask):
    dilated_mask = dilate_mask(mask)
    noshadow_avg = np.mean(noshadowimg[dilated_mask != 1])
    shadow_avg = np.mean(shadowimg[dilated_mask != 1])
    dmask = expand_to_three_channel(dilated_mask)
    dmask = np.moveaxis(dmask, 0, 2)
    gamma = noshadow_avg / shadow_avg
    noshadow_adjusted = noshadowimg * gamma
    noshadow_adjusted[noshadow_adjusted > 1] = 1
    return noshadow_adjusted