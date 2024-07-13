from typing import Union, Optional

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Focal_Loss(nn.Module):
    '''Multi-class Focal loss implementation from https://arxiv.org/abs/1708.02002'''
    def __init__(self, gamma:Union[float, int]=2, alpha:Union[float, int, None]=None, reduction:str='mean'):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = alpha
        self.reduction = reduction

    def forward(self, pred_logits:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
        """
        pred_logits [torch.Tensor]: the prediction logits in the format of [N, C]
        targets [torch.Tensor]: the true label in the format of [N, ]
        """
        log_pt = F.log_softmax(pred_logits, dim=-1)
        ce_loss = F.nll_loss(log_pt, targets, self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        fl_loss = (1-pt)**self.gamma * ce_loss
        if self.reduction == 'sum':
            fl_loss = fl_loss.sum()
        if self.reduction == 'mean':
            fl_loss = fl_loss.mean()
        return fl_loss


class QuadraticOrdinal_Loss(nn.Module):
    def __init__(self, num_classes:int, device, reduction:str='mean', weight:Union[float, int]=5, neg_log:bool=False):
        ''' Ordinal loss implementation inspired from'''
        super(QuadraticOrdinal_Loss, self).__init__()
        self.num_classes = torch.tensor(num_classes, device=device)
        self.idx_arr = torch.arange(num_classes, dtype=torch.float, device=device)
        self.reduction = reduction
        self.weight = weight
        self.neg_log = neg_log

    def forward(self, pred_logits:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
        """
        pred_logits [torch.Tensor]: the prediction logits in the format of [N, C]
        targets [torch.Tensor]: the true label in the format of [N, ]
        """
        if self.neg_log:
            p = -F.log_softmax(1-pred_logits, dim=-1)
        else:
            p = F.softmax(pred_logits, dim=-1)
        w = (targets.contiguous().view(targets.size(0), 1) - self.idx_arr)**2 / (self.num_classes - 1)**2
        loss = self.weight * (w * p).sum(dim=-1)

        if self.reduction == 'mean':
            loss = loss.mean()

        if self.reduction == 'sum':
            loss = loss.sum()

        return loss
    
class MergeCriterion_Loss(nn.Module):
    def __init__(self, loss_lst:list):
        '''
        loss_list [list]: the list of loss function for example:
        >>> criterion = MergeCriterion_Loss([nn.CrossEntropyLoss(), QuadraticOrdinal_Loss(configs['num_classes'], device=device)])
        '''
        super(MergeCriterion_Loss, self).__init__()
        self.__loss_lst = loss_lst

    @property
    def loss_list(self):
        return self.__loss_lst
    
    def forward(self, pred_logits:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
        overall_loss = 0
        for loss_fn in self.__loss_lst:
            loss_val = loss_fn(pred_logits, targets)
            overall_loss += loss_val
        return overall_loss
