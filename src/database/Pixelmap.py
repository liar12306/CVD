import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import shutil
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from PIL import Image
from src import config

import torchvision.transforms.functional as transF
import random


# from skimage import io, transform

class PixelMap_fold_STmap(Dataset):
    def __init__(self, root_dir, Training=True, transform=None, VerticalFlip=False, video_length=300):

        self.train = Training
        self.root_dir = root_dir
        self.transform = transform

        self.video_length = video_length
        self.VerticalFlip = VerticalFlip
        self.data_list = []
        train_file = config.PROJECT_ROOT + config.train_data_paths
        with open(train_file, 'r') as f:
            for line in f.readlines():
                self.data_list.append(line.strip('\n'))
        if Training:
            self.data_list = self.data_list[0:int(len(self.data_list)*0.8)]
        else:
            self.data_list = self.data_list[int(len(self.data_list) * 0.8):-1]

    def __len__(self):

        return len(self.data_list)

    def __getitem__(self, idx):

        data_path = self.data_list[0]
        data = np.load(data_path, allow_pickle=True).item()

        feature_map1 = Image.fromarray(data["rgb_map"])
        feature_map2 = Image.fromarray(data["yuv_map"])

        if self.VerticalFlip:
            if random.random() < 0.5:
                feature_map1 = transF.vflip(feature_map1)
                feature_map2 = transF.vflip(feature_map2)

        if self.transform:
            feature_map1 = self.transform(feature_map1)
            feature_map2 = self.transform(feature_map2)

        feature_map = torch.cat((feature_map1, feature_map2), dim=0)
        bpm = torch.tensor(data['bpm'], dtype=torch.float).unsqueeze(0)

        fps = torch.tensor(data['fps'],dtype=torch.float).unsqueeze(0)

        bvp = torch.tensor(data['bvp'][0], dtype=torch.float).unsqueeze(0)

        return (feature_map, bpm, fps, bvp, idx)
