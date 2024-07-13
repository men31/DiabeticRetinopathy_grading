import torch
from torch import nn
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_f1_score

import tqdm


def train(model, train_loader, optimizer, criterion, device):

    model = model.to(device)
    model.train()
    iteration = 0
    running_loss = 0.0
    running_corrects = 0.0
    running_f1 = 0.0

    for sample in tqdm.tqdm(train_loader):

        inputs = sample['image'].to(device, non_blocking=True)
        labels = sample['label'].to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        actual = labels

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == actual)
        running_f1 += multiclass_f1_score(preds, actual, num_classes=5)

        iteration += 1

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    epoch_f1 = running_f1 / iteration

    return {'loss':epoch_loss, 'acc':epoch_acc, 'f1': epoch_f1}


def test(model, val_loader, criterion, device):

    model = model.to(device)
    model.eval()
    iteration = 0
    running_loss = 0.0
    running_corrects = 0.0
    running_f1 = 0.0

    with torch.no_grad():
        for sample in tqdm.tqdm(val_loader):
            inputs = sample['image'].to(device)
            labels = sample['label'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            # _, actual = torch.max(labels, 1)
            actual = labels

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == actual)
            running_f1 += multiclass_f1_score(preds, actual, num_classes=5)

            iteration += 1

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)
    epoch_f1 = running_f1 / iteration
    
    return {'loss':epoch_loss, 'acc':epoch_acc.item(), 'f1': epoch_f1.item()}


