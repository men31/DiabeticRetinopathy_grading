import pathlib
import sys

import torch
from torch import nn
from torch.optim import lr_scheduler
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsampler import ImbalancedDatasetSampler

from model import *
from APTOS_dataset import APTOS_dataset
from utils.utils import EarlyStopping, BestCheckpoint, get_label_weights, BestSateCheckpoint
from behaviours import train, test
from utils.loss import Focal_Loss, MergeCriterion_Loss, QuadraticOrdinal_Loss

torch.manual_seed(2000)


def training(model_dict, configs, logging_path='logs'):
    ''' Device setup'''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    ''' Data setup'''
    img_dir = r'D:\Aj_Aof_Work\OCT_Disease\DATASET\APTOS2019_V2\images'
    train_label_dir = r'D:\Aj_Aof_Work\OCT_Disease\DATASET\APTOS2019_V2\labels\train.csv'
    val_label_dir = r'D:\Aj_Aof_Work\OCT_Disease\DATASET\APTOS2019_V2\labels\val.csv'

    transforms_train = v2.Compose([
        v2.ToImage(),
        v2.RandomEqualize(),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(degrees=(0, 180)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.41333666, 0.22077196, 0.0735625], 
                    [0.23908101, 0.13196332, 0.04935341]), 
        v2.Resize(model_dict['input_size']),
        ])

    transforms_val = v2.Compose([
        v2.ToTensor(),
        v2.Normalize([0.41333666, 0.22077196, 0.0735625], 
                     [0.23908101, 0.13196332, 0.04935341]), 
        v2.Resize(model_dict['input_size'])])

    train_data = APTOS_dataset(img_dir, train_label_dir, transform=transforms_train)
    val_data = APTOS_dataset(img_dir, val_label_dir, transform=transforms_val)
    train_loader = DataLoader(train_data, batch_size=configs['batch_size'], 
                              sampler=ImbalancedDatasetSampler(train_data), num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=configs['batch_size'], shuffle=False, num_workers=6)

    ''' Model setup '''
    model = model_dict['model']
    print('Re-check activation:', model.activation)
    # model = torch.compile(model, mode='reduce-overhead')

    ''' Hyperparameter setup '''
    num_epochs = configs['epochs']
    weights = get_label_weights(train_data)
    criterion = nn.CrossEntropyLoss(weight=weights).to(device=device)
    # criterion = Focal_Loss()
    # criterion = MergeCriterion_Loss([nn.CrossEntropyLoss(), QuadraticOrdinal_Loss(configs['num_classes'], device=device)])
    # optimizer = torch.optim.Adam(model.parameters(), configs['lr'])
    optimizer = torch.optim.AdamW(model.parameters(), configs['lr'])

    ''' Checkpoint setup '''
    ## Early stopping
    # early_stop = EarlyStopping(logging_path=logging_path, patience=10, mode='min')
    # Save optimal model
    # best_checkpoint = BestCheckpoint(logging_path=logging_path, mode='min')
    best_checkpoint = BestSateCheckpoint(logging_path=logging_path, mode='min')
    # Logging
    writer = SummaryWriter(logging_path)
    

    ''' Training loop '''
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        train_metric = train(model, train_loader, optimizer, criterion, device)
        val_metric = test(model, val_loader, criterion, device)

        # post-processing
        # if early_stop(model, val_metric['loss']):
        #     writer.close()
        #     sys.exit()
        # best_checkpoint(model, val_metric['loss'])
        save_dict = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_metric['loss']}
        best_checkpoint(save_dict, val_metric['loss'])

        for key, value in train_metric.items():
            writer.add_scalar('train_' + key, value, epoch)
        for key, value in val_metric.items():
            writer.add_scalar('val_' + key, value, epoch)
        # scheduler.step()
    writer.close()


def main(logging_path='logs'):
    ''' Training configurations '''
    configs = {'lr': 0.0002,
                'batch_size': 32,
                'epochs': 100, 
                'num_classes': 5}

    ''' Define the model '''
    # models_dict = {'resnet50':{'model': ResNet50(configs['num_classes'], transfer=True),
    #                           'input_size': (224, 224)}, 
    #                 'vgg19':{'model': VGG19(configs['num_classes'], transfer=True),
    #                           'input_size': (224, 224)},
    #                 'densenet161': {'model': DenseNet161(configs['num_classes'], transfer=True),
    #                           'input_size': (256, 256)}, 
    #                 'inception_v3': {'model': Inception_V3(configs['num_classes'], transfer=True),
    #                           'input_size': (299, 299)}, 
    #                 'swin_s': {'model': Swin_S(configs['num_classes'], transfer=True),
    #                           'input_size': (224, 224)},}
    models_dict = {'vgg19':{'model': VGG19(configs['num_classes'], transfer=True),
                    'input_size': (224, 224)}}
    
    for model_name, a_model_dict in models_dict.items():
        print('Model name:', model_name)
        training(a_model_dict, configs, logging_path=f'{model_name}_Imsampler_WCEloss_v1')


if __name__ =='__main__':
    tensorboard_dir = 'test_1'
    logging_path = pathlib.Path(tensorboard_dir)
    main(logging_path=logging_path) 
