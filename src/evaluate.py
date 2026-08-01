import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from model import get_model
from dataloader import test_loader
from config import DEVICE, BEST_MODEL_PATH


model = get_model().to(DEVICE)

model.load_state_dict(
    torch.load(BEST_MODEL_PATH, map_location=DEVICE)
)

model.eval()

all_predictions = []
all_labels = []


with torch.no_grad():

    for videos, labels in test_loader:

        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(videos)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


accuracy = accuracy_score(all_labels, all_predictions)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)


print("\n--- TEST RESULTS ---")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)