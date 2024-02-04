import torch
import torch.nn as nn
import os
from collections import OrderedDict

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:] if 'module.' in k else k
            # handle refactored layers
            name = k.replace('dowsample', 'downsample') if 'dowsample' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_arch(opt):
    from model import UNet,ShadowFormer
    arch = opt.arch

    print('You choose '+arch+'...')
    if arch == 'UNet':
        model_restoration = UNet(dim=opt.embed_dim)
    elif arch == 'ShadowFormer':
        model_restoration = ShadowFormer(
            img_size = opt.train_ps,
            embed_dim = opt.embed_dim,
            win_size = opt.win_size,
            token_projection = opt.token_projection,
            token_mlp = opt.token_mlp,
            use_log = opt.log_transform,
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