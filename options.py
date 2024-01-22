import argparse

PNG_DIVISOR = 255
TIFF_DIVISOR = 65535
EXR_DIVISOR = 1

RUN_LABEL = "sRGB"

PNG_DIR = "/work/SuperResolutionData/ShadowRemovalData/ISTD_Dataset"
PNG_LOG_DIR = "PNG"

LOG_DIR = f"/work/SuperResolutionData/ShadowRemovalResults/ShadowFormer2"
PLINEAR_LOG_DIR = "pseudolinear"
PLOG_DIR = "pseudolog"
PSEUDO_NO_NORM = "pseudo_no_normf"

PRETRAINED_WTS_DIR = f"/work/SuperResolutionData/ShadowRemovalResults/ShadowFormer2/{RUN_LABEL}/ShadowFormer_ISTD/models/model_best.pth"

# MAKE SURE TO CHANGE
# LOG DIR
# PRETRAINED WEIGHTS DIR
# DATA DIR

class LoadOptions():
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

class Options():
    """docstring for Options"""

    def __init__(self, description = None):

        parser = argparse.ArgumentParser(description=description)
        
        # global settings
        parser.add_argument('--run_label', type=str, default=RUN_LABEL, help='label for logs')
        parser.add_argument('--batch_size', type=int, default=4, help='batch size')
        parser.add_argument('--nepoch', type=int, default=500, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=1, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=1, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default='ISTD')
        parser.add_argument('--pretrain_weights', type=str, default=PRETRAINED_WTS_DIR,
                            help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')  # previous default: 0.01
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay') # L2 regularization, previous default: 0.01
        parser.add_argument('--gpu', type=str, default='0', help='GPUs')
        parser.add_argument('--arch', type=str, default='ShadowFormer', help='architecture')
        parser.add_argument('--mode', type=str, default='shadow', help='image restoration mode')

        # args for saving
        parser.add_argument('--save_dir', type=str, default=f'{LOG_DIR}/{RUN_LABEL}', help='save dir')
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
        parser.add_argument('--target_adjust', action='store_true', default=False, help='Adjust target colors to match ground truth')

        # parse arguments and copy into self
        parser.parse_args(namespace=self)
    
    def load_opts(self):
        '''Subset of options for loading images'''
        return LoadOptions(
            divisor=self.img_divisor, 
            linear_transform=self.linear_transform, 
            log_transform=self.log_transform,
            target_adjust=self.target_adjust
        )
