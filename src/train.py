import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CHECKPOINT_DIR
)
from dataloader import train_loader, val_loader
from model import get_model
from utils import calculate_accuracy
import random
import numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = get_model().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=2
)

best_val_accuracy = 0.0
patience = 5
epochs_without_improvement = 0

def validate(model, dataloader, criterion, device):

    model.eval()

    running_loss = 0.0
    running_accuracy = 0.0

    with torch.no_grad():

        for videos, labels in dataloader:

            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)

            loss = criterion(outputs, labels)

            accuracy = calculate_accuracy(outputs, labels)

            running_loss += loss.item()
            running_accuracy += accuracy

    avg_loss = running_loss / len(dataloader)
    avg_accuracy = running_accuracy / len(dataloader)

    return avg_loss, avg_accuracy

#training loop
for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    running_accuracy = 0.0

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

    val_loss, val_accuracy = validate(
    model,
    val_loader,
    criterion,
    DEVICE
    )

    print(
    f"\nEpoch [{epoch+1}/{EPOCHS}]"
    )

    print(
        f"Train Loss : {epoch_loss:.4f}"
    )

    print(
        f"Train Acc  : {epoch_accuracy:.4f}"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Acc    : {val_accuracy:.4f}"
    )


    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0
        torch.save(
            model.state_dict(),
            os.path.join(CHECKPOINT_DIR, "balanced_best_model.pth")
        )
        print("Best model saved.")

    else:
        epochs_without_improvement += 1
        print(
            f"No improvement for {epochs_without_improvement} epoch(s)."
        )

    torch.save(
    model.state_dict(),
    os.path.join(CHECKPOINT_DIR, "balanced_last_model.pth")
    )

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Learning Rate : {current_lr:.6f}")
    
    if epochs_without_improvement >= patience:
        print("\nEarly stopping triggered.")
        break