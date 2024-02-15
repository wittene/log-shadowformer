import os
import sys
import json
import warnings

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx
from losses import CharbonnierLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

import utils
from utils import Checkpoint
from utils.loader import get_training_data, get_validation_data

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'./auxiliary/'))
print(dir_name)

######### Options ###########
from options import TrainOptions
opt = TrainOptions(description='image denoising')
print(vars(opt))

output_opts = opt.output_opts
load_opts = opt.load_opts
img_opts_train = {
    'patch_size': opt.train_ps
}

MAX_VAL = 1 if not load_opts.log_transform else np.log(load_opts.log_range)

if opt.resume and not os.path.exists(output_opts.weights_latest):
    warnings.warn('--resume flag set to True, but cannot find weights. Setting --resume to False and training from scratch...')
    opt.resume = False


######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
torch.backends.cudnn.benchmark = True
# from piqa import SSIM
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)


######### Logs dir ########### 
print("LOG DIR: ", output_opts.log_dir)
if not os.path.exists(output_opts.log_dir):
    os.makedirs(output_opts.log_dir)
logname = os.path.join(output_opts.log_dir, output_opts.run_label+'.txt')
losslogname = os.path.join(output_opts.log_dir, output_opts.run_label+'.json')
print("Now time is : ", datetime.datetime.now().isoformat())

utils.mkdir(output_opts.model_dir)
utils.mkdir(output_opts.residuals_dir)



# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)



######### Resume ###########
checkpoint = None
start_epoch = 1
if opt.resume:
    # load checkpoint
    checkpoint = utils.load_checkpoint(output_opts.weights_latest, map_location='cuda')
    start_epoch = checkpoint.epoch + 1
    # correct the log
    with open(losslogname,'r') as f:
        d = json.load(f)
    to_delete = []
    for k in d["train"].keys():
        if int(k) >= start_epoch:
            to_delete.append(k)
    for k in to_delete:
        del d["train"][k]
    to_delete = []
    for k in d["val"].keys():
        if int(k) >= start_epoch:
            to_delete.append(k)
    for k in to_delete:
        del d["val"][k]
    with open(losslogname,'w') as f:
        json.dump(d, f)



######### Model ###########
model_restoration = utils.get_arch(opt)
model_restoration.cuda()
if checkpoint:
    checkpoint.load_model(model_restoration)


with open(logname,'a') as f:
    if opt.resume:
        f.write('\n'*4)
    f.write(str(vars(opt))+'\n')
    if opt.resume:
        f.write('\n'*2)
    if not opt.resume:
        f.write(str(model_restoration)+'\n')

######### DataParallel ###########
model_restoration = torch.nn.DataParallel(model_restoration)



######### Optimizer ###########
# eps = .1
# if opt.optimizer.lower() == 'adam':
#     optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=eps, weight_decay=opt.weight_decay)
# elif opt.optimizer.lower() == 'adamw':
#         optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=eps, weight_decay=opt.weight_decay)
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")

if checkpoint:
    checkpoint.load_optim(optimizer)

    

#     lr = utils.load_optim(optimizer, path_chk_rest)
#
#     for p in optimizer.param_groups: p['lr'] = lr
#     warmup = False
#     new_lr = lr
#     print('------------------------------------------------------------------------------')
#     print("==> Resuming Training with learning rate:",new_lr)
#     print('------------------------------------------------------------------------------')
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)

# ######### Scheduler ###########
if checkpoint:
    print("Using cosine strategy!")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)
elif opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    if checkpoint:
        checkpoint.load_scheduler(scheduler)
    optimizer.step()
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    if checkpoint:
        checkpoint.load_scheduler(scheduler)
    optimizer.step()
    scheduler.step()


######### Loss ###########
criterion = CharbonnierLoss().cuda()

######### DataLoader ###########
print('===> Loading datasets')
train_dataset = get_training_data(opt.train_dir, load_opts=load_opts, img_opts=img_opts_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.train_workers, pin_memory=True, drop_last=False)

val_dataset = get_validation_data(opt.val_dir, load_opts=load_opts)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

print("Size of training set: ", len(train_dataset),", size of validation set: ", len(val_dataset))

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_psnr  = checkpoint.best_psnr if checkpoint else 0
best_epoch = checkpoint.best_epoch if checkpoint else 0
best_iter  = checkpoint.best_iter if checkpoint else 0
eval_now = 1000
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
ii=0
index = 0
epoch_losses = []
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    epoch_ssim_loss = 0
    for i, data in enumerate(train_loader, 0): 
        # zero_grad
        index += 1
        optimizer.zero_grad()
        target = data[0].cuda()
        input_ = data[1].cuda()
        mask = data[2].cuda()
        
        if epoch > 5 and not (load_opts.linear_transform or load_opts.log_transform):
            target, input_, mask = utils.MixUp_AUG().aug(target, input_, mask)
        # E-Edit {
        with torch.cuda.amp.autocast():
            # forward pass -- returns images in input space
            restored, residual = model_restoration(input_, mask)
            restored = torch.clamp(restored,0,MAX_VAL)
            # compute loss in input space
            loss = criterion(restored, target)
        # } E-Edit
        loss_scaler(loss, optimizer, parameters=model_restoration.parameters())
        epoch_loss +=loss.item()

        #### Evaluation ####
        if (index+1)%eval_now==0 and i>0:

            eval_shadow_rmse = 0
            eval_nonshadow_rmse = 0
            eval_rmse = 0
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    mask = data_val[2].cuda()
                    filenames = data_val[3]
                    # E-Edit {
                    with torch.cuda.amp.autocast():
                        # forward pass
                        restored, residual = model_restoration(input_, mask)
                        restored = torch.clamp(restored,0,MAX_VAL)
                    # } E-Edit
                    # E-Edit {
                    # model returns image in input space (log-space), convert output and target to linear for evaluation
                    if load_opts.log_transform:
                        restored = utils.log_to_linear(restored, log_range=load_opts.log_range)
                        target = utils.log_to_linear(target, log_range=load_opts.log_range)
                        # mask = torch.multiply(mask, np.log(load_opts.log_range))
                    if load_opts.linear_transform:
                        restored = utils.apply_srgb(restored, max_val=MAX_VAL)
                        target = utils.apply_srgb(target, max_val=MAX_VAL)
                    # } E-Edit
                    # compute PSNR for batch
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())

                psnr_val_rgb = sum(psnr_val_rgb)/len(val_loader)
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    utils.save_checkpoint(
                        Checkpoint(epoch=epoch, best_psnr=best_psnr, best_epoch=best_epoch, best_iter=best_iter, model=model_restoration, optimizer=optimizer, scheduler=scheduler),
                        output_opts.weights_best
                    )
                    print(f'SAVED TO: {output_opts.weights_best}')
                print("[Ep %d it %d\t PSNR : %.4f] " % (epoch, i, psnr_val_rgb))
                with open(logname,'a') as f:
                    f.write("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
                        % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr)+'\n')
                model_restoration.train()
                torch.cuda.empty_cache()
    eval_loss = 0
    if epoch > 1 and (epoch < 10 or epoch % 3 == 0 or epoch == opt.nepoch):
        # calculate validation loss at set epoch intervals during training
        with torch.no_grad():
            eval_loss = 0
            model_restoration.eval()
            for ii, data_val in enumerate((val_loader), 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()
                mask = data_val[2].cuda()
                filenames = data_val[3]
                # E-Edit {
                with torch.cuda.amp.autocast():
                    # forward pass
                    restored, residual = model_restoration(input_, mask)
                    restored = torch.clamp(restored,0,MAX_VAL)
                # compute loss
                eval_loss += criterion(restored, target)
                # Output residual
                if opt.save_residuals and (epoch < 10 or epoch % 10 == 0 or epoch == opt.nepoch):
                    residuals_sub_dir = os.path.join(output_opts.residuals_dir, f"epoch_{epoch}")
                    utils.mkdir(residuals_sub_dir)
                    residual = residual.cpu().detach().numpy()
                    residual = np.clip(residual, 0, MAX_VAL)
                    # Enable to save residual in sRGB, otherwise it will save in input space
                    if load_opts.linear_transform:
                        residual = utils.apply_srgb(residual, max_val=MAX_VAL)
                    utils.save_img((residual*255.0).astype(np.ubyte), os.path.join(residuals_sub_dir, filenames[0]))
                # } E-Edit
            model_restoration.train()
            torch.cuda.empty_cache()
            with open(losslogname,'r') as f:
                d = json.load(f)
                d["val"][epoch] = eval_loss.item()
            with open(losslogname,'w') as f:
                json.dump(d, f)
         
    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss,scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    print(f"SAVING LOG TO: {logname}")
    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')
    if not os.path.isfile(losslogname):
        with open(losslogname,'w') as f:
            json.dump({"train": {}, "val": {}}, f)
    with open(losslogname,'r') as f:
        d = json.load(f)
        d["train"][epoch] = epoch_loss
    with open(losslogname,'w') as f:
        json.dump(d, f)

    utils.save_checkpoint(
        Checkpoint(epoch=epoch, best_psnr=best_psnr, best_epoch=best_epoch, best_iter=best_iter, model=model_restoration, optimizer=optimizer, scheduler=scheduler),
        os.path.join(output_opts.model_dir,"model_latest.pth")
    )

    if epoch%opt.checkpoint == 0:
        utils.save_checkpoint(
            Checkpoint(epoch=epoch, best_psnr=best_psnr, best_epoch=best_epoch, best_iter=best_iter, model=model_restoration, optimizer=optimizer, scheduler=scheduler),
            os.path.join(output_opts.model_dir, f"model_epoch_{epoch}.pth")
        )

print("Now time is : ",datetime.datetime.now().isoformat())
