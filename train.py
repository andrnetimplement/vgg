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

import os

data_dir = 'imagenet/ILSVRC/Data/CLS-LOC/train'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

BATCH_SIZE = 1024
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

num_classes = 1000
model19 = torch.load('vgg_models/model19_cifar10.pth')
model19.classifier[6] = nn.Linear(4096, 1000)


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model19 = nn.DataParallel(model19)
model19.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model19.parameters(), lr=0.001, momentum=0.9)

def train(model, epochs):
    for epoch in range(epochs):
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
            if i % 100 == 99:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # Оценка на тестовой выборке
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
        
        print(f'Accuracy on test set: {100 * correct / total:.2f}%')
        
train(model19, epochs=100)

split_indices = {
    'train': train_dataset.indices,
    'test': test_dataset.indices
}

# Сохранение индексов
torch.save(split_indices, 'vgg_models/dataset_split.pt')
torch.save(model19, 'vgg_models/model19_epoch100.pt')