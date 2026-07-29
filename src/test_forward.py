import torch

from dataloader import train_loader
from model import get_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model().to(device)
model.eval()

videos, labels = next(iter(train_loader))

videos = videos.to(device)

with torch.no_grad():
    outputs = model(videos)

print(f"Input shape  : {videos.shape}")
print(f"Output shape : {outputs.shape}")

print("\nModel outputs:")
print(outputs)