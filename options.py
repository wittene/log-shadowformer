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

# MAKE SURE TO CHANGE
# LOG DIR
# PRETRAINED WEIGHTS DIR
# DATA DIR

class LoadOptions():
    '''Options when loading an image'''
    def __init__(self, divisor=PNG_DIVISOR, linear_transform=False, log_transform=False, target_adjust=False, log_range=LOG_RANGE):
        # Normalization constant
        self.divisor = divisor
        # Flag for linear transform
        self.linear_transform = linear_transform
        # Flag for log transform
        self.log_transform = log_transform
        # Flag for target color adjustment
        self.target_adjust = target_adjust
        # Upper bound for log values
        self.log_range = log_range

class OutputOptions():
    '''Options for program output'''
    def __init__(self, arch, env, run_label, save_dir=None, pretrain_weights=None) -> None:
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
        self.pretrain_weights = pretrain_weights if pretrain_weights else os.path.join(self.model_dir, "model_latest.pth")
        # Path to residuals
        self.residuals_dir = os.path.join(self.log_dir, 'residuals')

class Options():
    """Options for training"""

    def __init__(self, description = None):
        self.__init_parser_args__(description=description)
        self.__init_load_opts__()
        self.__init_output_opts__()
    
    
    
    ##################################################
    # CONSTRUCTOR HELPERS
    
    def __init_parser_args__(self, description = None):
        '''Initialize argparse and add values to self'''

        parser = argparse.ArgumentParser(description=description)

        parser.add_argument('--run_label', type=str, help='label for logs')
        
        # global settings
        parser.add_argument('--batch_size', type=int, default=4, help='batch size')
        parser.add_argument('--nepoch', type=int, default=500, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=1, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=1, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default='ISTD')
        parser.add_argument('--pretrain_weights', type=str, default=None, help='path of pretrained_weights, calculated if unset')
        parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')  # previous default: 0.01
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay') # L2 regularization, previous default: 0.01
        parser.add_argument('--gpu', type=str, default='0', help='GPUs')
        parser.add_argument('--arch', type=str, default='ShadowFormer', help='architecture')
        parser.add_argument('--mode', type=str, default='shadow', help='image restoration mode')

        # args for saving
        parser.add_argument('--save_dir', type=str, default=None, help='save dir, calculated if unset')
        parser.add_argument('--save_images', action='store_true', default=False)
        parser.add_argument('--env', type=str, default='_ISTD', help='env')
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')
        parser.add_argument('--save_residuals', action='store_true', default=False, help='Save residuals during training')

        # args for Uformer
        parser.add_argument('--norm_layer', type=str, default='nn.LayerNorm', help='normalize layer in transformer')
        parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
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
        parser.add_argument('--train_ps', type=int, default=320, help='patch size of training sample')
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument('--train_dir', type=str, default=f'{PNG_DIR}/train', help='dir of train data')
        parser.add_argument('--val_dir', type=str, default=f'{PNG_DIR}/test', help='dir of train data')
        parser.add_argument('--warmup', action='store_true', default=True, help='warmup')
        parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup')

        # args for linear and log training
        parser.add_argument('--img_divisor', type=float, default=PNG_DIVISOR, help='value to scale images to [0, 1]') # Just leave it default.
        parser.add_argument('--linear_transform', action='store_true', default=False, help='Transform to pseudolinear') # get pseudolinear data
        parser.add_argument('--log_transform', action='store_true', default=False, help='Transform to pseudolog') # must call both flags, --linear_transform and --log_transform
        parser.add_argument('--log_range', type=int, default=LOG_RANGE, help='Upper bound of log value range')
        parser.add_argument('--target_adjust', action='store_true', default=False, help='Adjust target colors to match ground truth')

        # parse arguments and copy into self
        parser.parse_args(namespace=self)
    
    def __init_load_opts__(self):
        '''Subset of options for loading images'''
        self.load_opts = LoadOptions(
            divisor=self.img_divisor, 
            linear_transform=self.linear_transform, 
            log_transform=self.log_transform,
            target_adjust=self.target_adjust,
            log_range=self.log_range
        )
        # Ensure consistency
        self.img_divisor = self.load_opts.img_divisor
        self.linear_transform = self.load_opts.linear_transform
        self.log_transform = self.load_opts.log_transform
        self.target_adjust = self.load_opts.target_adjust
        self.log_range = self.load_opts.log_range
    
    def __init_output_opts__(self):
        '''Subset of options for saving output'''
        self.output_opts = OutputOptions(
            arch=self.arch,
            env=self.env,
            run_label=self.run_label,
            save_dir=self.save_dir,
            pretrain_weights=self.pretrain_weights
        )
        # Ensure consistency
        self.arch=self.output_opts.arch
        self.env=self.output_opts.env
        self.run_label=self.output_opts.run_label
        self.save_dir=self.output_opts.save_dir
        self.pretrain_weights=self.output_opts.pretrain_weights
