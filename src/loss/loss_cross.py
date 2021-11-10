import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, Function
import os
import shutil
import numpy as np
import scipy.io as sio
from scipy.stats import norm


class Cross_loss(nn.Module):
    def __init__(self, lambda_cvd=10):
        super(Cross_loss, self).__init__()

        self.lossfunc_HR = nn.L1Loss()
        self.lossfunc_feat = nn.L1Loss()

        self.lambda_cvd = lambda_cvd

    def forward(self, feat_p, feat_n, hr, feat_pes_p1, feat_pes_n1, hr_pes1, idx1, feat_pes_p2, feat_pes_n2, hr_pes2,
                idx2, gt):
        loss_hr1 = self.lossfunc_HR(hr_pes1, gt[idx1, :])
        loss_hr2 = self.lossfunc_HR(hr_pes2, gt[idx2, :])

        loss_fhr1 = self.lossfunc_feat(feat_pes_p1, feat_p[idx1, :, :, :])
        loss_fhr2 = self.lossfunc_feat(feat_pes_p2, feat_p[idx2, :, :, :])

        loss_fn1 = self.lossfunc_feat(feat_pes_n2, feat_n[idx1, :, :, :])
        loss_fn2 = self.lossfunc_feat(feat_pes_n1, feat_n[idx2, :, :, :])

        loss_hr_dis1 = self.lossfunc_HR(hr_pes1, hr[idx1, :])
        loss_hr_dis2 = self.lossfunc_HR(hr_pes2, hr[idx2, :])

        # loss = self.lambda_hr * (loss_hr1 + loss_hr2) / 2 + self.lambda_fhr * (loss_fhr1 + loss_fhr2) / 2 + self.lambda_fn * (loss_fn1 + loss_fn2) / 2
        loss = loss_hr_dis1 + loss_hr_dis2 + (loss_fn2 + loss_fn1) * self.lambda_cvd + (
                    loss_fhr1 + loss_fhr1) * self.lambda_cvd
        return loss, loss_hr1, loss_hr2, loss_fhr1, loss_fhr2, loss_fn1, loss_fn2, loss_hr_dis1, loss_hr_dis2
