"""
@Encoding:      UTF-8
@File:          cifar10_resnet18_trainer.py

@Introduction:  train resnet18 model for Cifar10 dataset, with configs given in edadaltocg/resnet18_cifar10
@Author:        Kaiming Zhu
@Date:          2024/6/6 23:59
@Reference:     [1] https://github.com/edadaltocg/detectors/blob/64c9eee87de99f44d217462998e08127d1d5c62e/scripts/train.py
                [2] https://huggingface.co/edadaltocg/resnet18_cifar10/tree/main
"""

import os
from pathlib import Path
import time
from typing import Optional, Tuple, Callable

import timm
import timm.data
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm


target_folder = str(Path(__file__).parent.absolute()) + "/cifar10/"
epochs = 300
data_path = "~/data"
device = "cuda:4"
normalize: Callable = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.201]
)


def save_model(epoch: int, model: nn.Module, target_device="cpu"):    
    model = model.to(target_device)
    torch.save(model.state_dict(), target_folder + f"epoch_{epoch}.pt")
    model = model.to(device)


def evaluate_acc_loss_for(model: nn.Module, loader: DataLoader, criterion: Optional[nn.Module] = None) -> Tuple[float, float]:
    if criterion is None:
        criterion = nn.CrossEntropyLoss().to(device)
    
    if model.training:
        model.eval()

    with torch.no_grad():
        length = torch.zeros(1).to(device)
        sum_correct_amount = torch.zeros(1).to(device)
        sum_loss = torch.zeros(1).to(device)
        for data, label in loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            inferred_label = torch.argmax(output, dim=-1)

            losses = criterion(output, label)
            is_correct_list = (inferred_label == label)
            sum_correct_amount = sum_correct_amount + torch.sum(is_correct_list)
            sum_loss = sum_loss + torch.sum(losses)
            length = length + data.size(0)

        avg_acc: float = (sum_correct_amount / length).item()
        avg_loss: float = (sum_loss / length).item()

        return avg_acc, avg_loss


def make_model() -> nn.Module:
    model: nn.Module = timm.models.create_model(model_name="resnet18", pretrained=False)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 10)
    return model


def make_optimizer_scheduler(model: nn.Module):
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=3, threshold=0.001, mode="max")
    return optimizer, scheduler


if __name__ == "__main__":
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    model: nn.Module = make_model().to(device=device)
    trainset = datasets.CIFAR10(
        root=data_path, 
        train=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ]), 
        download=True
    )
    testset = datasets.CIFAR10(
        root=data_path, 
        train=False, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]), 
        download=True
    )
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer, scheduler = make_optimizer_scheduler(model=model)
    
    accs = []
    losses = []
    best_acc = 0
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(epochs):
        start_time = time.time()
        
        if not model.training:
            model.train()
        
        for i, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        
        acc, loss = evaluate_acc_loss_for(model=model, loader=test_loader, criterion=criterion)
        accs.append(acc)
        losses.append(losses)
        scheduler.step(acc)
        
        if best_loss > loss:
            best_loss = loss
            best_acc = acc
            best_epoch = epoch
        
        save_model(epoch=epoch, model=model)
        print(f"epoch {epoch}/{epochs}: loss={loss:.5f}, acc={acc:.5f}, best_acc={best_acc:.5f}, time={time.time() - start_time:.2f}s")

    torch.save({"accs": accs, "losses": losses, "best_epoch": best_epoch}, target_folder + "metrics.pt")
