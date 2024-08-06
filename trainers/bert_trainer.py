import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW
from tqdm import tqdm
import os
from utils import calculate_accuracies


def train_bert_classifier(model, train_dataset, val_dataset, num_iterations, learning_rate, batch_size, save_path):
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set up the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Set up the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    iteration = 0
    best_val_accuracy = 0.0

    while iteration < num_iterations:
        for batch in tqdm(train_loader, desc=f"Iteration {iteration + 1}/{num_iterations}"):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % 100 == 0:
                # Evaluate on validation set
                avg_acc, worst_group_acc, group_accuracies = calculate_accuracies(model, val_loader, device)
                print(f"Iteration {iteration}, Validation Accuracy: {avg_acc:.4f}, Worst-group Accuracy: {worst_group_acc:.4f}")
                print(f"Group Accuracies: {group_accuracies}")

                # Save the best model (you might want to consider using worst-group accuracy here)
                if avg_acc > best_val_accuracy:
                    best_val_accuracy = avg_acc
                    torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))

                model.train()

            if iteration >= num_iterations:
                break

    # Save the final model
    torch.save(model.state_dict(), os.path.join(save_path, "final_model.pth"))
    print("Training completed. Final model saved.")

# Example usage:
# model = BertClassifier(config)
# train_dataset = YourTrainDataset()
# val_dataset = YourValDataset()
# train_bert_classifier(model, train_dataset, val_dataset, num_iterations=10000, learning_rate=2e-5, batch_size=32, save_path="./models")




def finetune_bert_classifier(model, train_dataset, val_dataset, num_iterations, learning_rate, batch_size, save_path):
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Freeze the BERT layers
    for param in model.bert.parameters():
        param.requires_grad = False

    # Set up the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set up the optimizer (only for classifier parameters)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Set up the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    iteration = 0
    best_val_accuracy = 0.0
    train_iter = iter(train_loader)

    with tqdm(total=num_iterations, desc="Training") as pbar:
        while iteration < num_iterations:
            try:
                inputs, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, labels = next(train_iter)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration += 1
            pbar.update(1)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if iteration % 100 == 0:
                # Evaluate on validation set
                avg_acc, worst_group_acc, group_accuracies = calculate_accuracies(model, val_loader, device)
                print(f"\nIteration {iteration}, Validation Accuracy: {avg_acc:.4f}, Worst-group Accuracy: {worst_group_acc:.4f}")
                print(f"Group Accuracies: {group_accuracies}")

                # Save the best model (you might want to consider using worst-group accuracy here)
                if avg_acc > best_val_accuracy:
                    best_val_accuracy = avg_acc
                    torch.save(model.state_dict(), os.path.join(save_path, "best_finetuned_model.pth"))

                model.train()

    # Save the final model
    torch.save(model.state_dict(), os.path.join(save_path, "final_finetuned_model.pth"))
    print("Fine-tuning completed. Final model saved.")

# Example usage:
# model = BertClassifier(config)
# train_dataset = YourTrainDataset()
# val_dataset = YourValDataset()
# finetune_bert_classifier(model, train_dataset, val_dataset, num_iterations=10000, learning_rate=2e-5, batch_size=32, save_path="./models")