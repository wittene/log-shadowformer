import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from collections import OrderedDict

class Checkpoint:
    '''Helper for saving/loading training progress'''
    def __init__(self,
                 epoch: int, 
                 best_psnr: float,
                 best_epoch: int,
                 best_iter: int,
                 model: nn.Module, 
                 optimizer: optim.Optimizer, 
                 scheduler: optim.lr_scheduler._LRScheduler
                 ):
        # Last completed epoch
        self.epoch = epoch
        # Best metrics so far
        self.best_psnr  = best_psnr
        self.best_epoch = best_epoch
        self.best_iter  = best_iter
        # Model state dict
        if isinstance(model, nn.Module):
            self.model = model.state_dict()
        else:
            self.model = model
        # Optimizer state dict
        if isinstance(optimizer, optim.Optimizer):
            self.optimizer = optimizer.state_dict()
        else:
            self.optimizer = optimizer
        # LR Scheduler state dict
        if isinstance(scheduler, optim.lr_scheduler._LRScheduler):
            self.scheduler = scheduler.state_dict()
        else:
            self.scheduler = scheduler
    
    # Helpers for loading state dicts
    def load_model(self, model: nn.Module):
        try:
            model.load_state_dict(self.model)
        except:
            # try to handle known parsing issues
            try:
                # handle refactored layers in order to load an older model
                layer_refactor_state_dict = OrderedDict()
                for k, v in self.model.items():
                    name = k.replace('dowsample', 'downsample') if 'dowsample' in k else k
                    layer_refactor_state_dict[name] = v
                model.load_state_dict(layer_refactor_state_dict)
            except:
                # also handle issue with module. prefix
                prefix_refactor_state_dict = OrderedDict()
                for k, v in layer_refactor_state_dict.items():
                    name = k[7:] if 'module.' in k else k
                    prefix_refactor_state_dict[name] = v
                model.load_state_dict(prefix_refactor_state_dict)
    
    def load_optim(self, optimizer: optim.Optimizer):
        if self.optimizer is not None:
            optimizer.load_state_dict(self.optimizer)
        else:
            warnings.warn('No optimizer to load.')
    
    def load_scheduler(self, scheduler: optim.lr_scheduler._LRScheduler):
        if self.scheduler is not None:
            scheduler.load_state_dict(self.scheduler)
        else:
            warnings.warn('No scheduler to load.')

    # Create dict for saving
    def to_dict(self):
        return {'epoch'     : self.epoch, 
                'best_psnr' : self.best_psnr,
                'best_epoch': self.best_epoch,
                'best_iter' : self.best_iter,
                'state_dict': self.model,
                'optimizer' : self.optimizer,
                'scheduler' : self.scheduler,
                }

    # Initialize from dict
    def from_dict(d: dict):
        return Checkpoint(
            epoch      = d['epoch'] if 'epoch' in d else 0,
            best_psnr  = d['best_psnr'] if 'best_psnr' in d else 0,
            best_epoch = d['best_epoch'] if 'best_epoch' in d else 0,
            best_iter  = d['best_iter'] if 'best_iter' in d else 0,
            model      = d['state_dict'],
            optimizer  = d['optimizer'] if 'optimizer' in d else None,
            scheduler  = d['scheduler'] if 'scheduler' in d else None,
        )



def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(checkpoint: Checkpoint, checkpoint_path: str):
    torch.save(checkpoint.to_dict(), checkpoint_path) 

def load_checkpoint(checkpoint_path: str, map_location='cpu') -> Checkpoint:
    checkpoint_dict = torch.load(checkpoint_path, map_location=map_location)
    return Checkpoint.from_dict(checkpoint_dict)

def get_arch(opt):
    from model import UNet,ShadowFormer
    arch = opt.arch

    print('You choose '+arch+'...')
    if arch == 'UNet':
        model_restoration = UNet(dim=opt.embed_dim)
    elif arch == 'ShadowFormer':
        model_restoration = ShadowFormer(
            img_size = opt.patch_size,
            embed_dim = opt.embed_dim,
            win_size = opt.win_size,
            token_projection = opt.token_projection,
            token_mlp = opt.token_mlp,
            log_range = opt.log_range,
            # DEFAULT PARAMS:
            # in_chans: int = 3,
            # depths: Any = [2, 2, 2, 2, 2, 2, 2, 2, 2],
            # num_heads: Any = [1, 2, 4, 8, 16, 16, 8, 4, 2],
            # mlp_ratio: float = 4,
            # qkv_bias: bool = True,
            # qk_scale: Any | None = None,
            # drop_rate: float = 0,
            # attn_drop_rate: float = 0,
            # drop_path_rate: float = 0.1,
            # norm_layer: Any = nn.LayerNorm,
            # patch_norm: bool = True,
            # use_checkpoint: bool = False,
            # se_layer: bool = True,
            # downsample: type[Downsample] = Downsample,
            # upsample: type[Upsample] = Upsample,
        )
    else:
        raise Exception("Arch error!")

    return model_restoration