import os, sys
import tqdm
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tbparse import SummaryReader
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, accuracy_score

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

''' The function '''

def test_model(model, loader, device):
    model = model.to(device)
    model.eval()
    pred_lst = []
    actual_lst = []
    with torch.no_grad():
        for sample in tqdm.tqdm(loader):
            inputs = sample['image'].to(device)
            labels = sample['label'].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            actual = labels
            actual_lst.extend(actual.detach().tolist())
            pred_lst.extend(preds.detach().tolist())
    return np.array(actual_lst), np.array(pred_lst)


''' The class '''

class Call_Model:
    def __init__(self, model_folder_dir:str):
        self.model_folder_dir = model_folder_dir
        self.__model_list = os.listdir(model_folder_dir)
        self.__model_dict = {idx:val for idx, val in enumerate(self.__model_list)}

    @property
    def model_list(self):
        return self.__model_dict

    def load_model(self, idx):
        return torch.load(os.path.join(self.model_folder_dir, self.__model_list[idx], 'best_model.pth'))
    
    def test_model(self, idx:int, loader, device, return_prob:bool=False):
        model = self.load_model(idx).to(device)
        model.eval()
        pred_lst = []
        actual_lst = []
        prob_lst = []
        # print(len(loader))
        with torch.no_grad():
            for sample in tqdm.tqdm(loader):
                # print(labels)
                inputs = sample['image'].to(device)
                labels = sample['label'].to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                # _, actual = torch.max(labels, 1)
                actual = labels
                # print('predict:', preds.detach().cpu().tolist())
                actual_lst.extend(actual.detach().tolist())
                pred_lst.extend(preds.detach().tolist())
                prob_lst.extend(F.softmax(torch.tensor(outputs)).detach().tolist())
        # print('L:', actual_lst)
        if return_prob:
            return np.array(actual_lst), np.array(pred_lst), np.array(prob_lst)
        return np.array(actual_lst), np.array(pred_lst)
    
    def roughly_evaluation_all_model(self, loader, device, interested_model=None):
        evaluation_df = pd.DataFrame(columns=['Name', 'QWK', 'F1 (macro)', 'F1 (micro)', 'AUCROC (macro)', 'AUCROC (micro)', 'ACC'])
        for idx, name in self.__model_dict.items():
            if 'best_model.pth' not in os.listdir(os.path.join(self.model_folder_dir, name)):
                continue
            if interested_model:
                if not name.startswith(interested_model):
                    continue
            actual_lst, pred_lst, prob_lst = self.test_model(idx, loader, device, return_prob=True)
            kappa = cohen_kappa_score(actual_lst, pred_lst, weights='quadratic')
            f1_macro = f1_score(actual_lst, pred_lst, average='macro')
            f1_micro = f1_score(actual_lst, pred_lst, average='micro')
            roc_macro = roc_auc_score(actual_lst, prob_lst, average='macro', multi_class='ovr')
            roc_micro = roc_auc_score(actual_lst, prob_lst, average='micro', multi_class='ovr')
            acc = accuracy_score(actual_lst, pred_lst)
            evaluation_df = pd.concat([evaluation_df, pd.DataFrame([[name, kappa, f1_macro, f1_micro, roc_macro, roc_micro, acc]], columns=evaluation_df.columns)], ignore_index=True)

        return evaluation_df
    

class Call_Logs:
    def __init__(self, logs_folder_dir:str):
        self.logs_folder_dir = logs_folder_dir
        self.__logs_list = os.listdir(logs_folder_dir)
        self.__logs_dict = {idx:val for idx, val in enumerate(self.__logs_list)}

    @property
    def logs_list(self):
        return self.__logs_dict

    def load_logs(self, idx:int):
        reader = SummaryReader(os.path.join(self.logs_folder_dir, self.__logs_list[idx]))
        return reader.scalars

    def plot_logs(self, num_col:int=5, keyword:list[str]=['loss']):
        plt.rcParams.update({'font.size':20})
        num_data = len(self.logs_list)
        num_row = num_data // num_col if num_data % num_col == 0 else num_data // num_col + 1
        for a_keyword in keyword:
            plt.figure(figsize=(num_col * 5, num_row * 4))
            for idx in range(num_data):
                a = self.load_logs(idx)
                plt.subplot(num_row, num_col, idx+1)
                sns.lineplot(x='step', y='value', data=a[a.tag.str.contains(a_keyword)], hue='tag')
                logs_split = self.logs_list[idx].split('_')
                title_name = f'{logs_split[0]}_{logs_split[-2]}'
                plt.title(title_name)
                plt.ylabel(a_keyword)
                plt.legend(loc=1, fontsize=15)
            plt.tight_layout()
        plt.show()

    def plot_logs_scalar(self, select_idx=None, keyword:list[str]=['val_loss']):
        plt.rcParams.update({'font.size':25})
        if select_idx is None:
            select_idx = range(len(self.logs_list))
        for a_keyword in keyword:
            plt.figure(figsize=(10, 8))
            for idx in select_idx:
                a = self.load_logs(idx)
                a = a[a['tag'] == a_keyword]
                logs_split = self.logs_list[idx].split('_')
                label_name = f'{logs_split[0]}_{logs_split[-2]}'
                plt.plot(a['step'], a['value'], label=label_name)
            # plt.title(a_keyword)
            plt.ylabel(a_keyword)
            plt.xlabel('step')
            plt.legend()
            plt.tight_layout()
        plt.show()
