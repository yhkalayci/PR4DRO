import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import os

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.d_out = num_classes
        
        # Replace the last fully connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

class ResNetFeaturizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.d_out = self.resnet.fc.in_features
        
        # Remove the last fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
    def forward(self, x):
        # x is expected to be of shape (batch_size, 3, height, width)
        outputs = self.resnet(x)
        # Flatten the output
        outputs = outputs.view(outputs.size(0), -1)
        return outputs
    

def extract_features(model, data_loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Extracting features"):
            inputs = inputs.to(device)
            feats = model(inputs)
            features.append(feats.cpu().numpy())
            labels.append(targets.numpy())
    return np.vstack(features), np.vstack(labels)
