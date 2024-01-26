import torch

def srgb_to_rgb(srgb_img, max_val=1):
  # Max val is an optional parameter to scale the image to [0, 1].
  # Images must be scaled to [0, 1].
  srgb_img = srgb_img / max_val
  low_mask = srgb_img <= 0.04045
  high_mask = srgb_img > 0.04045
  srgb_img[low_mask] /= 12.92
  srgb_img[high_mask] = (((srgb_img[high_mask]+ 0.055)/1.055)**(2.4))
  srgb_img[srgb_img > 1.0] = 1.0
  srgb_img[srgb_img< 0.0] = 0
  return srgb_img

def log_to_linear(log_img, log_range=65535):
  linear_img = torch.exp(log_img)
  linear_img = torch.div(linear_img, log_range)
  return linear_img
