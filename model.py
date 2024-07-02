import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

class VGG(nn.Module):
    def __init__(self, feautures, num_clasess=1000, init_weight=True):
        super(VGG, self).__init__()
        self.feautures = feautures
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            
        )
        
