"""
@Encoding:      UTF-8 
@File:          celeba_identity_resnet18_trainer.py

@Introduction:  Training CelebA-HQ Identity Classifier with specified resolution, in transfer learning style.
@Author:        Kaiming Zhu
@Date:          2025/3/26 15:44
@Reference:     [1] Na, Dongbin, Sangwoo Ji, and Jong Kim. "Unrestricted black-box adversarial attack using gan with
                    limited queries." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.
                [2] https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch/blob/main/CelebAMask_HQ_Dataset_Processing.ipynb
@Notes:         Please Install CelebAHQ Dataset, and generate Identity-Dataset by instructions in ref[2] before you run this scripts.
"""

import os
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from tqdm import tqdm


# configs for model training
epochs = 50
device = torch.device("cuda:2")
rescale_size = (224, 224)
normalize: Callable = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
)

# configs for dataset loading and result saving
dataset_path = "~/data/celeba-hq-identity"
model_save_folder = str(Path(__file__).parent.absolute()) + f"/celeba_hq_{rescale_size[0]}x{rescale_size[1]}/"


def save_model(model: nn.Module, epoch: Optional[int]=None, customized_name: Optional[str]=None, device="cpu"):    
    model = model.to("cpu")
    if epoch is not None:
        torch.save(model.state_dict(), model_save_folder + f"epoch_{epoch}.pt")
    elif customized_name is not None:
        torch.save(model.state_dict(), model_save_folder + customized_name)
    else:
        raise ValueError("Please specified name or epoch.")
    
    model = model.to(device)


def make_model(classes: int, epoch: Optional[int] = None, device="cpu"):
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, classes)
    
    if epoch is not None:
        state_dict = torch.load(model_save_folder + f"epoch_{epoch}.pt")
        model.load_state_dict(state_dict)
        
    model = model.to(device)
    return model


if __name__ == "__main__":
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
    
    train_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, 'train'), 
        transform=transforms.Compose([
            transforms.Resize(rescale_size),
            transforms.RandomHorizontalFlip(), # data augmentation
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    test_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, 'test'), 
        transform=transforms.Compose([
            transforms.Resize(rescale_size),
            transforms.ToTensor(),
            normalize,
        ])
    )
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128)
    
    model = make_model(classes=len(train_dataset.classes), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    train_accs = []
    train_losses = []
    test_accs = []
    test_losses = []
    best_acc = 0
    best_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        # Train Phase
        with tqdm(enumerate(train_dataloader), total=len(train_dataloader)) as loop:
            # Train Phase
            model.train()
            train_sum_loss = 0.
            train_correct = 0.
            train_amount = 0
            
            for i, (inputs, labels) in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward inputs and get output
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # get loss value and update the network weights
                loss.backward()
                optimizer.step()

                train_sum_loss += loss.item()
                train_correct += torch.sum(preds == labels.data)
                train_amount += inputs.size()[0]
                loop.set_description(f'Epoch {epoch} [Batch {i}/{len(train_dataloader)}]')
                loop.set_postfix({
                    "train loss": "{:.4f}".format(train_sum_loss / train_amount),
                    "train acc": "{:.3f}".format(train_correct / train_amount), 
                })
                
            train_accs.append((train_correct / train_amount).item())
            train_losses.append((train_sum_loss / train_amount).item())

        # Test Phase (Run Silently)
        model.eval()
        test_sum_loss = 0.
        test_correct = 0
        test_amount = 0
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                test_sum_loss += loss.item()
                test_correct += torch.sum(preds == labels.data)
                test_amount += inputs.size()[0]
            
        test_accs.append((test_correct / test_amount).item())
        test_losses.append((test_sum_loss / test_amount).item())
        
        # Scoring Phase (Run Silently)
        save_model(model=model, epoch=epoch, device=device)
        if best_loss > test_losses[-1]:
            best_loss = test_losses[-1]
            best_acc = test_accs[-1]
            best_epoch = epoch
    
    # Save metrics
    torch.save(
        obj={
            "train_accs": train_accs,
            "train_losses": train_losses,
            "test_accs": test_accs, 
            "test_losses": test_losses, 
            "best_epoch": best_epoch
        }, 
        f=model_save_folder + "metrics.pt"
    )
    
    # Reload for the best model
    best_model = make_model(classes=len(train_dataset.classes), epoch=best_epoch, device=device)
    save_model(model=best_model, customized_name="best.pt", device=device)
    print(f"Training Finished, Best Epoch Located at {best_epoch} with test acc {best_acc}")