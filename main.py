########################################################
# This is an example of the training and test procedure
# You need to adjust the training and test dataloader based on your data
# CopyRight @ Xuesong Niu
########################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import shutil
import sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import scipy.io as sio
import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR
import time
from tqdm import tqdm

sys.path.append('..')

from src.database.Pixelmap import PixelMap_fold_STmap

from src.model.model_disentangle import HR_disentangle_cross
from src.loss.loss_cross import Cross_loss
from src.loss.loss_r import Neg_Pearson
from src.loss.loss_SNR import SNR_loss
import numpy as np

batch_size_num = 2
epoch_num = 70
learning_rate = 0.0005
test_batch_size = 5

toTensor = transforms.ToTensor()
resize = transforms.Resize(size=(320, 320))

#######################################################

lambda_rec = 50
lambda_rppg = 2
lambda_cvd = 10

video_length = 300
########################################################################
### This is only a simple toy example dataloader (utils/database/PixelMap.py)
### This dataloader do not include the cross-validation division and training/test division.
### You need to adjust your dataloader based on your own data.
### parameter: root_dir: location of the MSTmaps
###            VerticalFlip: random vertical flip for data augmentation
########################################################################
train_dataset = PixelMap_fold_STmap(root_dir='./data/train_data/',
                                    Training=True, transform=transforms.Compose([resize, toTensor]), VerticalFlip=True,
                                    video_length=video_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size_num,
                          shuffle=True, num_workers=4)

test_dataset = PixelMap_fold_STmap(root_dir='./data/train_data/',
                                   Training=False, transform=transforms.Compose([resize, toTensor]), VerticalFlip=False,
                                   video_length=video_length)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                         shuffle=False, num_workers=4)
#########################################################################

#########################################################################
#########################################################################
net = HR_disentangle_cross()
if torch.cuda.is_available():
    net.cuda()
#########################################################################

lossfunc_HR = nn.L1Loss()
lossfunc_img = nn.L1Loss()
lossfunc_cross = Cross_loss(lambda_cvd=lambda_cvd)
lossfunc_ecg = Neg_Pearson(downsample_mode=0)
lossfunc_SNR = SNR_loss(clip_length=video_length, loss_type=7)

optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': 0.0005}])


def hr_error(ground_true, predict):
    return abs(predict - ground_true)


def rmse(loss):
    return np.sqrt(np.mean(loss ** 2))


def mae(loss):
    return np.mean(loss)


def mer(ground_true, loss):
    return np.mean(loss / ground_true) * 100


def std(loss, hr_mae):
    return np.sqrt(np.mean((loss - hr_mae) ** 2))


def r(ground_true, predict):
    g = ground_true - np.mean(ground_true)
    p = predict - np.mean(predict)
    return np.sum(g * p) / (np.sqrt(np.sum(g ** 2)) * np.sqrt(np.sum(p ** 2)))


def compute_criteria(target_hr_list, predicted_hr_list):
    hr_loss = hr_error(target_hr_list, predicted_hr_list)
    hr_mae = mae(hr_loss)
    hr_rmse = rmse(hr_loss)
    hr_mer = mer(target_hr_list, hr_loss)
    hr_std = std(hr_loss, hr_mae)
    pearson = r(target_hr_list, predicted_hr_list)

    return {"MAE": hr_mae, "RMSE": hr_rmse, "STD": hr_std, "MER": hr_mer, "r": pearson}


def train(epoch):
    start_time = time.time()
    net.train()
    train_loss = 0
    count = 0
    predict_hr = []
    gt_hr = []
    for (data, bpm, fps, bvp, idx) in tqdm(train_loader):
        count+=1
        data = Variable(data)
        bvp = Variable(bvp)
        bpm = Variable(bpm.view(-1, 1))
        fps = Variable(fps.view(-1, 1))
        data, bpm = data.cuda(), bpm.cuda()
        fps = fps.cuda()
        bvp = bvp.cuda()

        feat_p, feat_n, output, img_out, \
        feat_pes_p1, feat_pes_n1, hr_pes1, idx1, \
        feat_pes_p2, feat_pes_n2, hr_pes2, idx2, \
        ecg, ecg_pes_1, ecg_pes_2 = net(data)

        for idx in range(bpm.shape[0]):
            predict_hr.append(output[idx].item() * fps[idx].item() * 60 / video_length)
            gt_hr.append(bpm[idx].item() * fps[idx].item() * 60 / video_length)

        loss_hr = lossfunc_HR(output, bpm)
        loss_rec = lossfunc_img(data, img_out) * lambda_rec * 2
        loss_rppg = lossfunc_ecg(ecg, bvp) * lambda_rppg

        loss_SNR, tmp = lossfunc_SNR(ecg, bpm, fps, pred=output, flag=None)
        loss_pre = loss_hr + loss_rppg + loss_SNR

        loss_cvd, loss_hr1, loss_hr2, loss_fhr1, loss_fhr2, loss_fn1, loss_fn2, loss_hr_dis1, loss_hr_dis2 = lossfunc_cross(
            feat_p, feat_n, output,
            feat_pes_p1, feat_pes_n1,
            hr_pes1, idx1,
            feat_pes_p2, feat_pes_n2,
            hr_pes2, idx2, bpm)
        loss = loss_rec + loss_cvd + loss_pre

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_time = time.time()
    cost_time = end_time - start_time
    metrics = compute_criteria(np.array(gt_hr), np.array(predict_hr))
    print(f"\nFinished [Epoch: {epoch}/{epoch_num}]",
          "\nTraining Loss: {:.3f} |".format(train_loss/count),
          "MAE : {:.3f} |".format(metrics["MAE"]),
          "RMSE : {:.3f} |".format(metrics["RMSE"]),
          "STD : {:.3f} |".format(metrics["STD"]),
          "MER : {:.3f}% |".format(metrics["MER"]),
          "r : {:.3f} |".format(metrics["r"]),
          "time: {} m:{} s".format(int(cost_time % 60), int(cost_time / 60)))


def test():
    net.eval()
    test_loss = 0
    count = 0
    predict_hr = []
    gt_hr = []
    for (data, hr, fps, bvp, idx) in test_loader:
        count += 1
        data = Variable(data)
        hr = Variable(hr.view(-1, 1))
        fps = Variable(fps.view(-1, 1))
        data, hr = data.cuda(), hr.cuda()

        feat_hr, feat_n, output, img_out, feat_hrf1, feat_nf1, hrf1, idx1, feat_hrf2, feat_nf2, hrf2, idx2, ecg, ecg1, ecg2 = net(
            data)
        loss = lossfunc_HR(output, hr)
        for idx in range(hr.shape[0]):
            predict_hr.append(output[idx].item() * fps[idx].item() * 60 / video_length)
            gt_hr.append(hr[idx].item() * fps[idx].item() * 60 / video_length)

        test_loss += loss.item()
    metrics = compute_criteria(np.array(gt_hr), np.array(predict_hr))
    print("\nTest Loss: {:.3f} |".format(test_loss/count),
          "MAE : {:.3f} |".format(metrics["MAE"]),
          "RMSE : {:.3f} |".format(metrics["RMSE"]),
          "STD : {:.3f} |".format(metrics["STD"]),
          "MER : {:.3f}% |".format(metrics["MER"]),
          "r : {:.3f} |".format(metrics["r"])
          )


def run():
    begin_epoch = 1
    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.5)
    for epoch in range(begin_epoch, epoch_num + 1):
        if epoch > 20:
            train_dataset.transform = transforms.Compose([resize, toTensor])
            train_dataset.VerticalFlip = False

            train_loader = DataLoader(train_dataset, batch_size=batch_size_num,
                                      shuffle=True, num_workers=4)
        train(epoch)
        test()


if __name__ == "__main__":
    run()
    # it = iter(train_loader)
    # data = next(it)
    # print(data[0].shape)
