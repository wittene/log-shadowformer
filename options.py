import os
import torch

PNG_DIVISOR = 255
TIFF_DIVISOR = 65535
EXR_DIVISOR = 1

PNG_DIR = "/work/SuperResolutionData/ShadowRemovalData/ISTD_Dataset"
PNG_LOG_DIR = "PNG"

LOG_DIR = "/work/SuperResolutionData/ShadowRemovalResults/ShadowFormer"
PLINEAR_LOG_DIR = "ShadowFormer_pseudolinear"
PLOG_DIR = "ShadowFormer_pseudolog"
PSEUDO_NO_NORM = "pseudo_no_normf"

# MAKE SURE TO CHANGE
# LOG DIR
# PRETRAINED WEIGHTS DIR
# DATA DIR

class Options():
    """docstring for Options"""

    def __init__(self):
        pass

    def init(self, parser):
        # global settings
        parser.add_argument('--run_label', type=str, default='scaled_log', help='label for logs')
        parser.add_argument('--batch_size', type=int, default=4, help='batch size')
        parser.add_argument('--nepoch', type=int, default=500, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=1, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=1, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default='ISTD')
        parser.add_argument('--pretrain_weights', type=str, default="/work/SuperResolutionData/ShadowRemovalResults/ShadowFormer/ShadowFormer_pseudolog/ShadowFormer_istd/models/model_best.pth",
                            help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        # parser.add_argument('--lr_initial', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay') # L2 regularization
        # parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay') # L2 regularization
        parser.add_argument('--gpu', type=str, default='0', help='GPUs')
        parser.add_argument('--arch', type=str, default='ShadowFormer', help='architecture')
        parser.add_argument('--mode', type=str, default='shadow', help='image restoration mode')

        # args for saving
        parser.add_argument('--save_dir', type=str, default=f'{LOG_DIR}/scaled_log', help='save dir')
        parser.add_argument('--save_images', action='store_true', default=False)
        parser.add_argument('--env', type=str, default='_istd', help='env')
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')

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

        # args for linear and log traing

        # these are added by me

        parser.add_argument('--img_divisor', type=float, default=PNG_DIVISOR, help='value to scale images to [0, 1]') # Just leave it default.
        parser.add_argument('--linear_transform', action='store_true', default=False, help='Transform to pseudolinear') # get pseudolinear data
        parser.add_argument('--log_transform', action='store_true', default=False, help='Transform to pseudolog') # must call both flags, --linear_transform and --log_transform
        # don't use this, it won't work, delete all references
        parser.add_argument('--lr_finder', action='store_true', default=False, help='Use LR scheduler')
        parser.add_argument('--lr_low', type=float, default=0.00002, help='start value for lr_finder')
        parser.add_argument('--lr_high', type=float, default=0.2, help='end value for lr_finder')

        return parser
