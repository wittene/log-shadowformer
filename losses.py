import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
    
    # def forward(self, x, y):
    #     if self.log_loss:
    #         x = torch.exp(x)
    #         x = torch.div(x, 65535)
    #         y = torch.exp(y)
    #         y= torch.div(y, 65535)
    #         diff = torch.sub(x, y)
    #         loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
    #         x = torch.mul(x, 65535)
    #         x[x!=0] = torch.log(x[x!=0])
    #         y = torch.mul(y, 65535)
    #         y[y!=0] = torch.log(y[y!=0])
    #         return loss

    #     diff = x - y
    #     # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
    #     loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
    #     return loss

    # def forward(self, x, y):
    #     if self.log_loss:
    #         xForLoss = log_to_linear(x)
    #         yForLoss = log_to_linear(y)
    #         # diff = xForLoss - yForLoss
    #         diff = torch.sub(xForLoss, yForLoss)
    #         loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
    #         return loss

    #     diff = x - y
    #     # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
    #     loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
    #     return loss
    
    # def forward(self, x, y):
    #     # if self.log_loss:
    #     #     xForLoss = log_to_linear(x)
    #     #     yForLoss = log_to_linear(y)
    #     #     # diff = xForLoss - yForLoss
    #     #     diff = torch.sub(xForLoss, yForLoss)
    #     #     loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
    #     #     return loss


    #     diff = x - y
    #     # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
    #     loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
    #     return loss
