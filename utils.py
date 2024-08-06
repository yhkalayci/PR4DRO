import torch

def calculate_accuracies(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    group_correct = {}
    group_total = {}
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels[:, 0]).sum().item()
            
            for i, (pred, label) in enumerate(zip(predicted, labels)):
                group = label[1].item()
                if group not in group_correct:
                    group_correct[group] = 0
                    group_total[group] = 0
                group_total[group] += 1
                if pred == label[0]:
                    group_correct[group] += 1

    average_acc = correct / total
    group_accuracies = {g: group_correct[g] / group_total[g] for g in group_correct}
    worst_group_acc = min(group_accuracies.values())

    return average_acc, worst_group_acc, group_accuracies