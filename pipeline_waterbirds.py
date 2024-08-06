import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
from tqdm import tqdm
import os
from utils import calculate_accuracies

from dataloaders.waterbirds import WaterbirdsDataModule, WaterbirdsDataset
from proportional_representation import get_subsample_indices_v2
from trainers.resnet_trainer import train_resnet_classifier, finetune_resnet_classifier
from models.resnet import ResNetClassifier, ResNetFeaturizer, extract_features

import ssl
import urllib.request

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

def waterbirds_pipeline(demo=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the dataset
    data_module = WaterbirdsDataModule(root='data', batch_size=32)
    data_module.setup()
    
    if demo:
        # Subsample data randomly with probability 0.1
        def subsample_dataset(dataset, prob=0.1):
            num_samples = len(dataset)
            num_keep = int(num_samples * prob)
            return random_split(dataset, [num_keep, num_samples - num_keep])[0]
        
        data_module.train_dataset = subsample_dataset(data_module.train_dataset)
        data_module.val_dataset = subsample_dataset(data_module.val_dataset)
        data_module.test_dataset = subsample_dataset(data_module.test_dataset)
        
        num_iterations = 5  # Reduce number of iterations for demo
    else:
        num_iterations = 10
    
    # Train the initial model
    model = ResNetClassifier(num_classes=2).to(device)
    train_resnet_classifier(model, data_module.train_dataset, data_module.val_dataset, 
                            num_iterations=num_iterations, learning_rate=0.001, batch_size=32, 
                            save_path="./models")
    
    # Save the pretrained model
    torch.save(model.state_dict(), 'waterbird_pretrain.pt')
    
    # Extract features for all datasets
    featurizer = ResNetFeaturizer().to(device)
    featurizer.resnet.load_state_dict(model.resnet.state_dict(), strict=False)
    
    train_features, train_labels = extract_features(featurizer, DataLoader(data_module.train_dataset, batch_size=32), device)
    val_features, val_labels = extract_features(featurizer, DataLoader(data_module.val_dataset, batch_size=32), device)
    test_features, test_labels = extract_features(featurizer, DataLoader(data_module.test_dataset, batch_size=32), device)
    
    # Use proportional representation to find subsample indices
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5] if not demo else [0.1, 0.2]
    subsample_indices = get_subsample_indices_v2(test_features, train_features, ratios)
    
    # Finetune for each subset
    for ratio in ratios:
        # Create a subset of the training data
        subset_indices = subsample_indices[ratio]
        subset_dataset = Subset(data_module.train_dataset, subset_indices)
        
        # Load the pretrained model
        finetuned_model = ResNetClassifier(num_classes=2).to(device)
        finetuned_model.load_state_dict(torch.load('waterbird_pretrain.pt'))
        
        # Finetune the model
        finetune_resnet_classifier(finetuned_model, subset_dataset, data_module.val_dataset, 
                                   num_iterations=num_iterations, learning_rate=0.0001, batch_size=32, 
                                   save_path=f"./waterbirds_finetune_{ratio}")
    
    test_loader = DataLoader(data_module.test_dataset, batch_size=32)
    
    print("\nEvaluating models on test set:")
    
    # Evaluate pretrained model
    model.load_state_dict(torch.load('waterbird_pretrain.pt'))
    avg_acc, worst_group_acc, group_accuracies = calculate_accuracies(model, test_loader, device)
    print(f"Pretrained Model - Average Accuracy: {avg_acc:.4f}, Worst-group Accuracy: {worst_group_acc:.4f}")
    print(f"Group Accuracies: {group_accuracies}")

    # Evaluate finetuned models
    for ratio in ratios:
        model.load_state_dict(torch.load(f"./waterbirds_finetune_{ratio}/best_finetuned_model.pth"))
        avg_acc, worst_group_acc, group_accuracies = calculate_accuracies(model, test_loader, device)
        print(f"Finetuned Model (ratio {ratio}) - Average Accuracy: {avg_acc:.4f}, Worst-group Accuracy: {worst_group_acc:.4f}")
        print(f"Group Accuracies: {group_accuracies}")

if __name__ == "__main__":
    waterbirds_pipeline(demo=True)  # Set to False for full pipeline