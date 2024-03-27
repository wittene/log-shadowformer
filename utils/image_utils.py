import torch
import numpy as np
import pickle
import cv2
import rawpy
from skimage.color import rgb2lab
import matplotlib.pyplot as plt
from options import LoadOptions




##################################################
# CONVERT/TRANSFORM

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
    img = np.load(filepath, allow_pickle=True)
    return img

def load_imgs(clean_filename, noisy_filename, mask_filename, load_opts: LoadOptions = LoadOptions(), data_transforms = None):
    '''Load the shadow, non-shadow, and mask images -- with chosen transforms'''

    # load files -- np.array with shape (H, W, C)
    if load_opts.img_type == 'raw':
        load_img  = lambda fp: (rawpy.imread(fp).postprocess().astype(np.float32)) / load_opts.divisor
        load_mask = lambda fp: ((cv2.imread(fp, cv2.IMREAD_GRAYSCALE) != 0).astype(np.float32))
    else:
        load_img  = lambda fp: (cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype(np.float32)) / load_opts.divisor
        load_mask = lambda fp: ((cv2.imread(fp, cv2.IMREAD_GRAYSCALE) != 0).astype(np.float32))
    
    clean = load_img(clean_filename)
    noisy = load_img(noisy_filename)
    mask  = load_mask(mask_filename)

    clean = np.clip(clean, 0, 1)
    noisy = np.clip(noisy, 0, 1)

    if data_transforms is not None:
        clean, noisy, mask = data_transforms(clean, noisy, mask)
    
    return clean, noisy, mask

def save_img(img, filepath):
    img_copy = img
    if len(img.shape) == 4:  # try to squeeze batch dimension
        if img.shape[0] > 1:
            raise Exception('save_img only accepts a single image -- batch dimension must be 1')
        img_copy = img_copy.squeeze()
    if img.shape[-1] > 3:    # try to rearrange dims for imwrite
        img_copy = img_copy.transpose((1, 2, 0))
    if filepath.lower().endswith('.cr2') or filepath.lower().endswith('.tiff'):
        filepath = ''.join(filepath.split('.')[:-1]) + '.png'
    cv2.imwrite(filepath, cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))



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
