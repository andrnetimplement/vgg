import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms, models
from torch.nn.parallel import DataParallel
from model import VGG, vgg19
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, filename='training2.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')


# 2 gpu train
# Функция для сохранения чекпоинта
def save_checkpoint(model, optimizer, epoch, train_indices, test_indices, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_indices': train_indices,
        'test_indices': test_indices
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

# Функция для загрузки чекпоинта
def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_indices = checkpoint['train_indices']
    test_indices = checkpoint['test_indices']
    return model, optimizer, epoch, train_indices, test_indices

# Функция обучения
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            logging.info(f'Train Epoch: {epoch} [{i * len(inputs)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

# Test function
def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    logging.info(f'Accuracy on test set: {100 * correct / total:.2f}%')
    
BATCH_SIZE = 1024
num_epochs = 60
num_workers = 30
# Путь к датасету
data_dir = 'imagenet/ILSVRC/Data/CLS-LOC/train'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка и подготовка данных
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
full_dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform) 
criterion = torch.nn.CrossEntropyLoss()

start_epoch = 0
checkpoint_path = 'vgg_models/model19_checkpoint.pth'

# Проверка наличия чекпоинта
if os.path.exists(checkpoint_path):
    LEARNING_RATE = 0.001 # start from 0.01
    model19 = vgg19(pretrained=False)
    model19.classifier[6] = nn.Linear(4096, 1000)
    model19, optimizer, start_epoch, train_indices, test_indices = load_checkpoint(checkpoint_path, model19, optimizer)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model19 = nn.DataParallel(model19)
    model19.to(device)
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    for param_group in optimizer.param_groups:
        param_group['lr'] = LEARNING_RATE
        
    print(f"Resuming from epoch {start_epoch}, lr: {optimizer.param_groups[0]['lr']}")
else:
    # Разбиение на train и test
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    # берем предобученную на cifar10 модель
    model19 = torch.load('vgg_models/model19_cifar10.pth')
    model19.classifier[6] = nn.Linear(4096, 1000)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model19 = nn.DataParallel(model19)
    model19.to(device)
    optimizer = torch.optim.SGD(model19.parameters(), lr=0.01, momentum=0.9)
    
    print("Starting training from scratch")

# Обучение
for epoch in range(start_epoch, num_epochs):
    train(model19, device, train_loader, optimizer, epoch)

# Сохранение чекпоинта
save_checkpoint(model19, optimizer, epoch + 1,
                train_dataset.indices, test_dataset.indices,
                checkpoint_path)

test(model19, device, test_loader)