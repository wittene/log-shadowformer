import torch
import numpy as np
import warnings

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

def log_to_linear(log_img, log_range=None):

  if not log_range:
    log_range = 65535
    warnings.warn('Log range was unset in log_to_linear. This may be a mistake! Defaulting to 65535.')

  if isinstance(log_img, torch.Tensor):
    # scale from [0,1] to log range
    linear_img = torch.mul(log_img, np.log(log_range))
    # exponentiate
    linear_img = torch.exp(linear_img)
    # scale again
    linear_img = torch.div(linear_img, log_range)
    linear_img = torch.clamp(linear_img, 0, 1)
  
  else:
    # scale from [0,1] to log range
    linear_img = log_img * np.log(log_range)
    # exponentiate
    linear_img = np.exp(linear_img)
    # scale again
    linear_img /= log_range
    linear_img = np.clip(linear_img, 0, 1)
  
  return linear_img

def linear_to_log(linear_img, log_range=None):
  '''Convert linear image in range [0,1] to log image in range [0,1]'''

  if not log_range:
    log_range = 65535
    warnings.warn('Log range was unset in log_to_linear. This may be a mistake! Defaulting to 65535.')

  if isinstance(linear_img, torch.Tensor):
    # scale before taking the log
    log_img = torch.mul(linear_img, log_range)
    # take the log
    log_img[log_img != 0] = torch.log(log_img[log_img != 0])
    # scale to [0,1]
    log_img = torch.div(log_img, log_range)
  else:
    # scale before taking the log
    log_img = linear_img * log_range
    # take the log
    log_img[log_img != 0] = np.log(log_img[log_img != 0])
    # scale to [0,1]
    log_img /= np.log(log_range)
  
  return log_img
