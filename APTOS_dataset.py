import os
import torch
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class APTOS_dataset(Dataset):
    '''
    APTOS 2019 dataset
    
    Attributes:
    ----------
        - img_dir [str]: the directory path of image
        - label_dir [str]: the directory path of label file, defualt: *.csv
        - transform: for transformation function of torchvision, defualt: None
        - num_classes [int]: number of class for classification task
        - balancing [bool]: balancing the amount of data in each classes, defualt: False

    '''
    def __init__(self, img_dir:str, label_dir:str, transform=None, balancing:bool=False, num_classes:int=5):
        self.img_dir = img_dir
        self.transform = transform
        self.raw_label_df = pd.read_csv(label_dir)
        self.label_df = self.raw_label_df.copy()
        if num_classes == 2:
            self.label_df.iloc[:, 1] = self.label_df.iloc[:, 1].apply(lambda x: 1 if x > 0 else 0)
        elif num_classes == 4:
            self.label_df = self.label_df[self.label_df.iloc[:, 1] != 0].reset_index().drop(columns=['index'])
            
        if balancing:
            self.label_df = self.make_balance_label()

        self.unique_labels = np.sort(self.raw_label_df.iloc[:, 1].unique())

    def get_labels(self):
        return self.label_df.iloc[:, 1].to_list()
    
    def make_balance_label(self):
        uqe, counts = np.unique(self.label_df.iloc[:, 1].to_list(), return_counts=True)
        min_count = min(counts)
        select_idx = []
        for i_uqe in uqe:
            select_idx.extend(self.label_df[self.label_df.iloc[:, 1] == i_uqe][:min_count].index.to_list())
        return self.label_df.copy().iloc[select_idx]

    def __len__(self):
        return len(self.label_df.index)
    
    def __getitem__(self, idx:int) -> dict:
        y = self.label_df.iloc[idx, 1]
        img_name = os.path.join(self.img_dir, f'{self.label_df.iloc[idx, 0]}.png')
        img = io.imread(img_name, as_gray=False)
        if self.transform:
            img = self.transform(img)
        sample = {'image': img, 'label': y}
        return sample

