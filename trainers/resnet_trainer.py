import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import os
from utils import calculate_accuracies

def train_resnet_classifier(model, train_dataset, val_dataset, num_iterations, learning_rate, batch_size, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

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

            outputs = model(inputs)
            loss = criterion(outputs, labels[:, 0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration += 1
            pbar.update(1)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if iteration % 100 == 0:
                avg_acc, worst_group_acc, group_accuracies = calculate_accuracies(model, val_loader, device)
                print(f"\nIteration {iteration}, Validation Accuracy: {avg_acc:.4f}, Worst-group Accuracy: {worst_group_acc:.4f}")
                print(f"Group Accuracies: {group_accuracies}")

                if avg_acc > best_val_accuracy:
                    best_val_accuracy = avg_acc
                    torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))

                model.train()

    torch.save(model.state_dict(), os.path.join(save_path, "final_model.pth"))
    print("Training completed. Final model saved.")

def finetune_resnet_classifier(model, train_dataset, val_dataset, num_iterations, learning_rate, batch_size, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for param in model.resnet.parameters():
        param.requires_grad = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    iteration = 0
    best_val_accuracy = 0.0
    train_iter = iter(train_loader)

    with tqdm(total=num_iterations, desc="Fine-tuning") as pbar:
        while iteration < num_iterations:
            try:
                inputs, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, labels = next(train_iter)

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs[:, 0], labels[:, 0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration += 1
            pbar.update(1)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if iteration % 100 == 0:
                avg_acc, worst_group_acc, group_accuracies = calculate_accuracies(model, val_loader, device)
                print(f"\nIteration {iteration}, Validation Accuracy: {avg_acc:.4f}, Worst-group Accuracy: {worst_group_acc:.4f}")
                print(f"Group Accuracies: {group_accuracies}")

                if avg_acc > best_val_accuracy:
                    best_val_accuracy = avg_acc
                    torch.save(model.state_dict(), os.path.join(save_path, "best_finetuned_model.pth"))

                model.train()

    torch.save(model.state_dict(), os.path.join(save_path, "final_finetuned_model.pth"))
    print("Fine-tuning completed. Final model saved.")