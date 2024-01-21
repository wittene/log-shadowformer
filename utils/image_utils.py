import torch
import numpy as np
import pickle
import cv2
from skimage.color import rgb2lab
import matplotlib.pyplot as plt

##################################################
# CONSTANTS

MAX_LOG_VAL = 65535



##################################################
# CLASSES

class LoadOpts():
    '''Options when loading an image'''
    def __init__(self, divisor=255, linear_transform=False, log_transform=False, target_adjust=False):
        # Normalization constant
        self.divisor = divisor
        # Flag for linear transform
        self.linear_transform = linear_transform
        # Flag for log transform
        self.log_transform = log_transform
        # Flag for target color adjustment
        self.target_adjust = target_adjust



##################################################
# CONVERT/TRANSFORM

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

def apply_srgb(linear_img:np.array, max_val:int=1)->np.array:
  '''
  Apply srgb to a linear image.
  Input:
    linear_img: a linear image. Needs to be scaled to the range [0, 1].
    Either input a linear image in the correct range or use the optional
    max_val parameter to scale the image.
  Output:
    The image with srgb applied in range [0, 1] as datatype float32.
  '''
  linear_img /= max_val
  if np.max(linear_img) > 1 or np.min(linear_img) < 0:
      raise Exception("linear_img must be scaled to [0, 1]. Use max_val with the appropriate max_val to scale the image.")
  low_mask = linear_img <= 0.0031308
  high_mask = linear_img > 0.0031308
  linear_img[low_mask] *= 12.92
  linear_img[high_mask] = ((linear_img[high_mask]*1.055)**(1/2.4)) - 0.055
  linear_img[linear_img > 1.0] = 1.0
  linear_img[linear_img < 0.0] = 0
  return linear_img

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):

        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
            # image_numpy = image_numpy.convert('L')

        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    # image_numpy =
    return np.clip(image_numpy, 0, 255).astype(imtype)

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def dilate_mask(mask):
    kernel = np.ones((8,8), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    return dilation

# def yCbCr2rgb(input_im):
#     im_flat = input_im.contiguous().view(-1, 3).float()
#     mat = torch.tensor([[1.164, 1.164, 1.164],
#                        [0, -0.392, 2.017],
#                        [1.596, -0.813, 0]])
#     bias = torch.tensor([-16.0/255.0, -128.0/255.0, -128.0/255.0])
#     temp = (im_flat + bias).mm(mat)
#     out = temp.view(3, list(input_im.size())[1], list(input_im.size())[2])



##################################################
# FILE TYPE

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])



##################################################
# SAVE/LOAD

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

def load_img(filepath, load_opts: LoadOpts = LoadOpts()):
    img = cv2.cvtColor(cv2.imread(filepath, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    img = img / load_opts.divisor
    if load_opts.linear_transform:
        img = srgb_to_rgb(img)
    if load_opts.linear_transform and load_opts.log_transform:
        img *= 65535
        img[img!=0] = np.log(img[img!=0])
        img /= np.log(65535)
    if load_opts.log_transform and not load_opts.linear_transform:
        raise Exception("Cannot perform a log transform without a linear transform first.")
    return img

def load_val_img(filepath, load_opts: LoadOpts = LoadOpts()):
    img = cv2.cvtColor(cv2.imread(filepath, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    resized_img = img.astype(np.float32)
    resized_img = resized_img /load_opts.divisor
    if load_opts.linear_transform:
        img = srgb_to_rgb(img)
    # We're calculating loss in linear space.
    if load_opts.linear_transform and load_opts.log_transform:
        pass
    if load_opts.log_transform and not load_opts.linear_transform:
        raise Exception("Cannot perform a log transform without a linear transform first.")
    return resized_img

def load_mask(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # kernel = np.ones((8,8), np.uint8)
    # erosion = cv2.erode(img, kernel, iterations=1)
    # dilation = cv2.dilate(img, kernel, iterations=1)
    # contour = dilation - erosion
    img = img.astype(np.float32)
    # contour = contour.astype(np.float32)
    # contour = contour/255.
    img = img/255
    return img

def load_val_mask(filepath):
    img = cv2.imread(filepath, 0)
    resized_img = img
    # resized_img = cv2.resize(img, [256, 256], interpolation=cv2.INTER_AREA)
    resized_img = resized_img.astype(np.float32)
    resized_img = resized_img/255.
    return resized_img

def save_img(img, filepath):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)



##################################################
# EVALUATION
# Calculate in linear space

def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)

def calc_RMSE(real_img, fake_img):
    # convert to LAB color space
    real_lab = rgb2lab(real_img)
    fake_lab = rgb2lab(fake_img)
    return real_lab - fake_lab
