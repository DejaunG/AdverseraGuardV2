import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import time
import copy
import os
from sklearn.metrics import classification_report
import numpy as np
import multiprocessing
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def ensure_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Advanced data augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Lambda(ensure_rgb),
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Lambda(ensure_rgb),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def main():
    data_dir = 'dataset'
    batch_size = 16
    num_epochs = 1500  # Increased number of epochs

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}

    # Adjusted class weights
    class_counts = torch.tensor([158, 16, 16, 202])
    class_weights = 1. / (class_counts + 1)  # Adding 1 to avoid division by zero
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights * 3  # Increase the effect of class weights

    train_targets = torch.tensor(image_datasets['train'].targets)
    train_samples_weight = class_weights[train_targets]
    train_sampler = WeightedRandomSampler(train_samples_weight, len(train_samples_weight))

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, sampler=train_sampler, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0)
    }

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Use EfficientNet-B3 for better performance
    model = models.efficientnet_b3(pretrained=True)

    # Fine-tune more layers
    for param in list(model.parameters())[:-60]:  # Freeze all but the last 60 layers
        param.requires_grad = False

    # Modify the classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    model = train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=num_epochs)

    torch.save(model.state_dict(), 'optimized_adversera_model.pth')
    print("Model saved. Class names:", class_names)


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=1500):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train' and random.random() < 0.5:  # Apply mixup to 50% of batches
                        inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)
                        outputs = model(inputs)
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                print(classification_report(all_labels, all_preds, target_names=dataloaders['train'].dataset.classes,
                                            zero_division=1))

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()