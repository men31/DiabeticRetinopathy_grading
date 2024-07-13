from typing import Any, Union
import torch
import torchvision.transforms as transforms
from PIL import Image
from copy import deepcopy
import os
import tqdm
from sklearn.metrics import cohen_kappa_score, f1_score, jaccard_score
import numpy as np
import pandas as pd


def model_checkpoint(model, logging_path:str):
    print('Model checkpoint')
    torch.save(model, os.path.join(logging_path, 'best_model.pth'))

def state_checkpoint(save_dict:dict, logging_path:str):
    print('State checkpoint')
    torch.save(save_dict, os.path.join(logging_path, 'best_model.tar'))

def create_ema_model(model):
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return ema_model

def get_label_weights(dataset):
    _, counts = np.unique(dataset.get_labels(), return_counts=True)
    weights = 1 - counts / counts.sum()
    return torch.FloatTensor(weights)


class EarlyStopping:
    def __init__(self, patience: int=5, logging_path: str='logs', thershold: Union[float, int]=0, mode: str='min'):
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.thershold = thershold
        self.logging_path = logging_path
        self.mode = mode

    def __call__(self, model, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            model_checkpoint(model, self.logging_path)
        if self.best_loss - current_loss >= self.thershold and self.mode == 'min' :
            print('Mode:', self.mode.upper())
            print(f'Best loss: {self.best_loss} -> Current loss: {current_loss}')
            print('Found improvement in training: Save model')
            self.counter = 0
            self.best_loss = current_loss
            model_checkpoint(model, self.logging_path)
        elif current_loss - self.best_loss >= self.thershold and self.mode == 'max' :
            print('Mode:', self.mode.upper())
            print(f'Best loss: {self.best_loss} -> Current loss: {current_loss}')
            print('Found improvement in training: Save model')
            self.counter = 0
            self.best_loss = current_loss
            model_checkpoint(model, self.logging_path)
        else:
            self.counter += 1
            if self.counter > self.patience:
                print('Reach the limit of patience: Early stopping')
                return True
        return False
    

class BestCheckpoint:
    def __init__(self, logging_path: str='logs', thershold: Union[float, int]=0, mode: str='min'):
        self.best_loss = None
        self.counter = 0
        self.thershold = thershold
        self.logging_path = logging_path
        self.mode = mode

    def __call__(self, model, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            model_checkpoint(model, self.logging_path)
        if self.best_loss - current_loss >= self.thershold and self.mode == 'min' :
            print('Mode:', self.mode.upper())
            print(f'Best loss: {self.best_loss} -> Current loss: {current_loss}')
            print('Found improvement in training: Save model')
            self.counter = 0
            self.best_loss = current_loss
            model_checkpoint(model, self.logging_path)
        elif current_loss - self.best_loss >= self.thershold and self.mode == 'max' :
            print('Mode:', self.mode.upper())
            print(f'Best loss: {self.best_loss} -> Current loss: {current_loss}')
            print('Found improvement in training: Save model')
            self.counter = 0
            self.best_loss = current_loss
            model_checkpoint(model, self.logging_path)
        else:
            pass


class BestSateCheckpoint:
    def __init__(self, logging_path: str='logs', thershold: Union[float, int]=0, mode: str='min'):
        self.best_loss = None
        self.counter = 0
        self.thershold = thershold
        self.logging_path = logging_path
        self.mode = mode

    def __call__(self, save_dict:dict, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            state_checkpoint(save_dict, self.logging_path)
        if self.best_loss - current_loss >= self.thershold and self.mode == 'min' :
            print('Mode:', self.mode.upper())
            print(f'Best loss: {self.best_loss} -> Current loss: {current_loss}')
            print('Found improvement in training: Save model')
            self.counter = 0
            self.best_loss = current_loss
            state_checkpoint(save_dict, self.logging_path)
        elif current_loss - self.best_loss >= self.thershold and self.mode == 'max' :
            print('Mode:', self.mode.upper())
            print(f'Best loss: {self.best_loss} -> Current loss: {current_loss}')
            print('Found improvement in training: Save model')
            self.counter = 0
            self.best_loss = current_loss
            state_checkpoint(save_dict, self.logging_path)
        else:
            pass
    

class DataAugmentation:
    def __init__(self, size:int=224):
        RandomGaussianBlur = lambda p: transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))],
            p=p
        )
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=Image.BICUBIC),
            RandomGaussianBlur(0.4),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.299, 0.244, 0.255)),
        ])

    def __call__(self, x):
        all_images = [self.transform(x), self.transform(x)]
        return all_images


class TransformTwin:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        out_1 = self.transform(x)
        out_2 = self.transform(x)
        return out_1, out_2
    
