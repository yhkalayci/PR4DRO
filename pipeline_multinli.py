import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import os
from utils import calculate_accuracies

from dataloaders.multinli import MultiNLIDataModule, MultiNLIDataset
from proportional_representation import get_subsample_indices_v2
from trainers.bert_trainer import train_bert_classifier, finetune_bert_classifier
from models.bert import BertClassifier, BertFeaturizer, extract_features

import ssl
import urllib.request

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

def multinli_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the dataset
    data_module = MultiNLIDataModule(root='data', batch_size=32)
    data_module.setup()
    
    # Train the initial model
    model = BertClassifier(num_classes=3).to(device)  # MultiNLI has 3 classes
    train_bert_classifier(model, data_module.train_dataset, data_module.val_dataset, 
                          num_iterations=10, learning_rate=2e-5, batch_size=32, 
                          save_path="./models")
    
    # Save the pretrained model
    torch.save(model.state_dict(), 'multinli_pretrain.pt')
    
    # Extract features for all datasets
    featurizer = BertFeaturizer().to(device)
    featurizer.bert.load_state_dict(model.bert.state_dict(), strict=False)
    
    train_features, train_labels = extract_features(featurizer, data_module.train_dataloader(), device)
    val_features, val_labels = extract_features(featurizer, data_module.val_dataloader(), device)
    test_features, test_labels = extract_features(featurizer, data_module.test_dataloader(), device)
    
    # Use proportional representation to find subsample indices
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    subsample_indices = get_subsample_indices_v2(test_features, train_features, ratios)
    
    # Finetune for each subset
    for ratio in ratios:
        # Create a subset of the training data
        subset_indices = subsample_indices[ratio]
        subset_dataset = Subset(data_module.train_dataset, subset_indices)
        
        # Load the pretrained model
        finetuned_model = BertClassifier(num_classes=3).to(device)
        finetuned_model.load_state_dict(torch.load('multinli_pretrain.pt'))
        
        # Finetune the model
        finetune_bert_classifier(finetuned_model, subset_dataset, data_module.val_dataset, 
                                 num_iterations=10, learning_rate=2e-5, batch_size=32, 
                                 save_path=f"./multinli_finetune_{ratio}")
    
    test_loader = DataLoader(data_module.test_dataset, batch_size=32)
    
    print("\nEvaluating models on test set:")
    
    # Evaluate pretrained model
    model.load_state_dict(torch.load('multinli_pretrain.pt'))
    avg_acc, worst_group_acc, group_accuracies = calculate_accuracies(model, test_loader, device)
    print(f"Pretrained Model - Average Accuracy: {avg_acc:.4f}, Worst-group Accuracy: {worst_group_acc:.4f}")
    print(f"Group Accuracies: {group_accuracies}")
    
    # Evaluate finetuned models
    for ratio in ratios:
        model.load_state_dict(torch.load(f"./multinli_finetune_{ratio}/best_finetuned_model.pth"))
        avg_acc, worst_group_acc, group_accuracies = calculate_accuracies(model, test_loader, device)
        print(f"Finetuned Model (ratio {ratio}) - Average Accuracy: {avg_acc:.4f}, Worst-group Accuracy: {worst_group_acc:.4f}")
        print(f"Group Accuracies: {group_accuracies}")

if __name__ == "__main__":
    multinli_pipeline()