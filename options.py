import os
import argparse

PNG_DIVISOR = 255
TIFF_DIVISOR = 65535
EXR_DIVISOR = 1

LOG_RANGE = 65535

PNG_DIR = "/work/SuperResolutionData/ShadowRemovalData/ISTD_Dataset"
PNG_LOG_DIR = "PNG"

LOG_DIR = f"/work/SuperResolutionData/ShadowRemovalResults/ShadowFormer2"
PLINEAR_LOG_DIR = "pseudolinear"
PLOG_DIR = "pseudolog"
PSEUDO_NO_NORM = "pseudo_no_normf"

VALID_IMG_TYPES = {'srgb', 'raw'}
VALID_MOTION_TRANSFORMS = {'affine', ''}


# Helper classes

class LoadOptions():
    '''Options when loading an image'''
    def __init__(self, 
                 dataset='ISTD', 
                 divisor=PNG_DIVISOR, 
                 img_type='sRGB',
                 resize=None, patch_size=None,
                 linear_transform=False, log_transform=False, log_range=LOG_RANGE,
                 color_balance_aug=False, intensity_aug=False, target_adjust=False, motion_transform=''):
        # Dataset
        self.dataset = dataset
        # Normalization constant
        self.divisor = divisor
        # Image type: sRGB, raw
        assert img_type.lower() in VALID_IMG_TYPES
        self.img_type = img_type.lower()
        # Resize
        self.resize = resize
        if self.resize is not None and self.resize < 64:
            self.resize = None
        # Training patch size
        self.patch_size = patch_size
        # Flag for linear transform
        self.linear_transform = linear_transform
        if self.img_type == 'raw':
            self.linear_transform = False  # already linear
        # Flag for log transform
        self.log_transform = log_transform
        # Upper bound for log values
        self.log_range = log_range
        # Flags for color adjustments
        self.color_balance_aug = color_balance_aug
        self.intensity_aug = intensity_aug
        self.target_adjust = target_adjust
        # Try to load path to motion transforms file
        assert motion_transform.lower() in VALID_MOTION_TRANSFORMS
        self.motion_transform = motion_transform.lower()

    def update(self, 
               dataset=None,
               divisor=None, 
               img_type=None,
               resize=None,
               patch_size=None,
               linear_transform=None, 
               log_transform=None, 
               color_balance_aug=None,
               intensity_aug=None,
               target_adjust=None, 
               log_range=None,
               motion_transform=None,
               ):
        dataset=dataset if dataset is not None else self.dataset,
        divisor=divisor if divisor is not None else self.divisor, 
        img_type=img_type if img_type is not None else self.img_type,
        resize=resize,
        patch_size=patch_size if patch_size is not None else self.patch_size,
        linear_transform=linear_transform if linear_transform is not None else self.linear_transform, 
        log_transform=log_transform if log_transform is not None else self.log_transform,
        color_balance_aug=color_balance_aug if color_balance_aug is not None else self.color_balance_aug,
        intensity_aug=intensity_aug if intensity_aug is not None else self.intensity_aug,
        target_adjust=target_adjust if target_adjust is not None else self.target_adjust,
        log_range=log_range if log_range is not None else self.log_range,
        motion_transform=motion_transform if motion_transform is not None else self.motion_transform
        return self


class OutputOptions():
    '''Options for program output'''
    def __init__(self, arch, env, run_label, save_dir=None, weights_latest=None, weights_best=None) -> None:
        # Define the run
        self.arch = arch
        self.env = env
        self.run_label = run_label
        # Directory to save output
        self.save_dir = save_dir if save_dir else os.path.join(LOG_DIR, self.run_label)
        # Directory to save output logs
        self.log_dir = os.path.join(self.save_dir, self.arch+self.env)
        # Path to model and weights
        self.model_dir = os.path.join(self.log_dir, 'models')
        self.weights_latest = weights_latest if weights_latest else os.path.join(self.model_dir, "model_latest.pth")
        self.weights_best = weights_best if weights_best else os.path.join(self.model_dir, "model_best.pth")
        # Path to residuals
        self.residuals_dir = os.path.join(self.log_dir, 'residuals')
        # Path to results
        self.results_dir = os.path.join(self.log_dir, 'results')
        # Path to diff images
        self.diffs_dir = os.path.join(self.log_dir, 'diffs')


# Option classes

class ProgramOptions():
    '''Base class, to build program-specific options'''

    # Static helpers

    def __add_load_args__(parser: argparse.ArgumentParser):
        parser.add_argument('--dataset', type=str, default='ISTD', help='dataset to use for eval: ISTD, RawSR')
        parser.add_argument('--img_divisor', type=float, default=PNG_DIVISOR, help='value to scale images to [0, 1]')
        parser.add_argument('--img_type', type=str, default='sRGB', help='Input image type: sRGB, raw')
        parser.add_argument('--resize', type=int, default=None, help='Resize longest side to this size, if 0, use original resolution')
        parser.add_argument('--patch_size', type=int, default=320, help='For training, patch size of training sample')
        parser.add_argument('--linear_transform', action='store_true', default=False, help='Transform to pseudolinear') # get pseudolinear data
        parser.add_argument('--log_transform', action='store_true', default=False, help='Transform to pseudolog')
        parser.add_argument('--log_range', type=int, default=LOG_RANGE, help='Upper bound of values prior to log transform')
        parser.add_argument('--color_balance_aug', action='store_true', default=False, help='Color balance data augmentation')
        parser.add_argument('--intensity_aug', action='store_true', default=False, help='Intensity data augmentation')
        parser.add_argument('--target_adjust', action='store_true', default=False, help='Adjust target colors to match ground truth')
        parser.add_argument('--motion_transform', type=str, default='', help='Type of motion transform to apply to targets, must be pre-computed in dataset directory')

    def __add_output_args__(parser: argparse.ArgumentParser):
        parser.add_argument('--run_label', type=str, help='label for logs')
        parser.add_argument('--arch', type=str, default='ShadowFormer', help='architecture')
        parser.add_argument('--env', type=str, default='_ISTD', help='env')
        parser.add_argument('--save_dir', type=str, default=None, help='save dir, calculated if unset')
        parser.add_argument('--weights_latest', type=str, default=None, help='path of latest pretrained weights, calculated if unset')
        parser.add_argument('--weights_best', type=str, default=None, help='path of best pretrained weights, calculated if unset')


    # Common program option methods
    
    def __init_output_opts__(self):
        '''Subset of options for saving output'''
        self.output_opts = OutputOptions(
            arch=self.arch,
            env=self.env,
            run_label=self.run_label,
            save_dir=self.save_dir,
            weights_latest=self.weights_latest,
            weights_best=self.weights_best
        )
        # Ensure consistency
        self.arch=self.output_opts.arch
        self.env=self.output_opts.env
        self.run_label=self.output_opts.run_label
        self.save_dir=self.output_opts.save_dir
        self.weights_latest=self.output_opts.weights_latest
        self.weights_best=self.output_opts.weights_best
    
    def __init_load_opts__(self):
        '''Subset of options for loading images'''
        self.set_load_opts(LoadOptions(
            dataset=self.dataset,
            divisor=self.img_divisor, 
            img_type=self.img_type,
            resize=self.resize,
            patch_size=self.patch_size,
            linear_transform=self.linear_transform, 
            log_transform=self.log_transform,
            color_balance_aug=self.color_balance_aug,
            intensity_aug=self.intensity_aug,
            target_adjust=self.target_adjust,
            log_range=self.log_range,
            motion_transform=self.motion_transform
        ))
    
    def set_load_opts(self, load_opts: LoadOptions):
        self.load_opts = load_opts
        # Ensure consistency
        self.dataset = self.load_opts.dataset
        self.img_divisor = self.load_opts.divisor
        self.img_type = self.load_opts.img_type
        self.resize = self.load_opts.resize
        self.patch_size = self.load_opts.patch_size
        self.linear_transform = self.load_opts.linear_transform
        self.log_transform = self.load_opts.log_transform
        self.color_balance_aug = self.load_opts.color_balance_aug
        self.intensity_aug = self.load_opts.intensity_aug
        self.target_adjust = self.load_opts.target_adjust
        self.log_range = self.load_opts.log_range
        self.motion_transform = self.load_opts.motion_transform
    
    def update_load_opts(self, 
                      dataset=None,
                      divisor=None, 
                      img_type=None,
                      resize=None,
                      patch_size=None,
                      linear_transform=None, 
                      log_transform=None, 
                      color_balance_aug=None,
                      intensity_aug=None,
                      target_adjust=None, 
                      log_range=None,
                      motion_transform=None,
                      ):
        self.set_load_opts(LoadOptions(
            dataset=dataset if dataset is not None else self.dataset,
            divisor=divisor if divisor is not None else self.img_divisor, 
            img_type=img_type if img_type is not None else self.img_type,
            resize=resize,
            patch_size=patch_size if patch_size is not None else self.patch_size,
            linear_transform=linear_transform if linear_transform is not None else self.linear_transform, 
            log_transform=log_transform if log_transform is not None else self.log_transform,
            color_balance_aug=color_balance_aug if color_balance_aug is not None else self.color_balance_aug,
            intensity_aug=intensity_aug if intensity_aug is not None else self.intensity_aug,
            target_adjust=target_adjust if target_adjust is not None else self.target_adjust,
            log_range=log_range if log_range is not None else self.log_range,
            motion_transform=motion_transform if motion_transform is not None else self.motion_transform
        ))
        return self.load_opts



class TrainOptions(ProgramOptions):
    """Options for training"""

    def __init__(self, description = None):
        super(TrainOptions, self).__init__()
        self.__init_parser_args__(description=description)
        self.__init_load_opts__()
        self.__init_output_opts__()

    ##################################################
    # CONSTRUCTOR HELPERS
    
    def __init_parser_args__(self, description = None):
        '''Initialize argparse and add values to self'''

        parser = argparse.ArgumentParser(description=description)
        
        # global settings
        parser.add_argument('--batch_size', type=int, default=4, help='batch size')
        parser.add_argument('--nepoch', type=int, default=500, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=1, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=1, help='eval_dataloader workers')
        parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')  # previous default: 0.01
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay') # L2 regularization, previous default: 0.01
        parser.add_argument('--gpu', type=str, default='0', help='GPUs')

        # args for Uformer
        parser.add_argument('--norm_layer', type=str, default='nn.LayerNorm', help='normalize layer in transformer')
        parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdedding features')
        parser.add_argument('--win_size', type=int, default=10, help='window size of self-attention')
        parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
        parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
        parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')

        # args for vit
        parser.add_argument('--vit_dim', type=int, default=320, help='vit hidden_dim')
        parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
        parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
        parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
        parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
        parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
        parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
        parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

        # args for training
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument('--train_dir', type=str, default=f'{PNG_DIR}/train', help='dir of train data')
        parser.add_argument('--val_dir', type=str, default=f'{PNG_DIR}/test', help='dir of validation data')
        parser.add_argument('--warmup', action='store_true', default=True, help='warmup')
        parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup')

        # args for image transforms
        ProgramOptions.__add_load_args__(parser)

        # args for output
        ProgramOptions.__add_output_args__(parser)
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')
        parser.add_argument('--save_residuals', action='store_true', default=False, help='Save residuals during training')

        # parse arguments and copy into self
        parser.parse_args(namespace=self)
  

class TestOptions(ProgramOptions):
    """Options for testing"""

    def __init__(self, description = None):
        super(TestOptions, self).__init__()
        self.__init_parser_args__(description=description)
        self.__init_load_opts__()
        self.__init_output_opts__()

    ##################################################
    # CONSTRUCTOR HELPERS
    
    def __init_parser_args__(self, description = None):
        '''Initialize argparse and add values to self'''

        parser = argparse.ArgumentParser(description=description)

        # global settings
        parser.add_argument('--gpu', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
        parser.add_argument('--input_dir', default=f'{PNG_DIR}/test', type=str, help='directory of validation images')
        parser.add_argument('--batch_size', default=1, type=int, help='batch size for dataloader')
        parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
        parser.add_argument('--tile_overlap', type=int, default=0, help='Overlapping of different tiles')

        # args for UFormer
        parser.add_argument('--embed_dim', type=int, default=32, help='dim of embedding features')    
        parser.add_argument('--win_size', type=int, default=10, help='window size of self-attention')
        parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
        parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
        
        # args for vit
        parser.add_argument('--vit_dim', type=int, default=320, help='vit hidden_dim')
        parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
        parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
        parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
        parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
        parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
        parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
        parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

        # args for image transforms
        ProgramOptions.__add_load_args__(parser)

        # args for output
        ProgramOptions.__add_output_args__(parser)
        parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
        parser.add_argument('--save_residuals', action='store_true', default=False, help='Save residuals')
        parser.add_argument('--save_diffs', action='store_true', default=False, help='Save target-output difference images')
        parser.add_argument('--cal_metrics', action='store_true', help='Measure denoised images with GT')

        # parse arguments and copy into self
        parser.parse_args(namespace=self)
