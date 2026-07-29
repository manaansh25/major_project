import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY
)

from dataloader import train_loader
from model import get_model
from utils import calculate_accuracy

model = get_model().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    running_accuracy = 0.0

    from tqdm import tqdm
    for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(videos)

        loss = criterion(outputs, labels)

        accuracy = calculate_accuracy(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        running_accuracy += accuracy

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = running_accuracy / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy:.4f}"
    )