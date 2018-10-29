#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 18:02:14 2018

@author: nirmal
"""
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import os
from skimage import io
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

class CustomDatasetFromImages(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, img_path,transforms):
    
        self.tmp_df = pd.read_csv(csv_path)
        
        self.img_path = img_path
        self.transforms=transforms
        self.X_train = np.asarray(self.tmp_df.iloc[:, 0])
        self.y_train = np.asarray(self.tmp_df.iloc[:, 1])
        self.data_len = len(self.tmp_df.index)

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] )
      
        img = self.transforms(img)
        
        p=[self.y_train[index]]
        label = torch.LongTensor(p)
        return img, label

    def __len__(self):
        return self.data_len

