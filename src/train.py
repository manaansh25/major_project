import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

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
import pandas as pd

SEED = 42

MODEL_NAME = "s3d" #change here to change the model architecture, options are: r2plus1d18, mc3_18, r3d_18, s3d
# parser = argparse.ArgumentParser()
# parser.add_argument("--model", default="r3d18")
# args = parser.parse_args()

# MODEL_NAME = args.model

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = get_model(MODEL_NAME).to(DEVICE)

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

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

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
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)


    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0
        torch.save(
            model.state_dict(),
            os.path.join(CHECKPOINT_DIR,
            f"{MODEL_NAME}_best_model.pth")
        )
        print("Best model saved.")

    else:
        epochs_without_improvement += 1
        print(
            f"No improvement for {epochs_without_improvement} epoch(s)."
        )

    torch.save(
    model.state_dict(),
    os.path.join(
    CHECKPOINT_DIR,
    f"{MODEL_NAME}_last_model.pth")
    )

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Learning Rate : {current_lr:.6f}")
    
    if epochs_without_improvement >= patience:
        print("\nEarly stopping triggered.")
        break


history_df = pd.DataFrame({
    "epoch": range(1, len(train_losses) + 1),
    "train_loss": train_losses,
    "train_accuracy": train_accuracies,
    "val_loss": val_losses,
    "val_accuracy": val_accuracies
})

history_df.to_csv(
    f"results/{MODEL_NAME}_training_history.csv",
    index=False
)

print(
    f"\nTraining history saved to results/{MODEL_NAME}_training_history.csv"
)