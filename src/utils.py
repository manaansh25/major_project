import torch

def calculate_accuracy(outputs, labels):
    predictions = torch.argmax(outputs, dim=1)

    correct = (predictions == labels).sum().item()
    accuracy = correct / labels.size(0)

    return accuracy