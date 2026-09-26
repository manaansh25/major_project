import os
import random
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from dataset import ViolenceDataset
from torch.utils.data import DataLoader
from model import get_model

from config import DEVICE


# ==============================
# REPRODUCIBILITY
# ==============================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ==============================
# SETTINGS
# ==============================

MODELS = [
    "r3d18",
    "r2plus1d18",
    "mc3_18",
    "s3d"
]

CORRUPTIONS = [
    None,
    "gaussian",
    "motion",
    "brightness",
    "gaussian_blur",
    "mixed"
]

CORRUPTION_PROBABILITY = 1.0

RESULTS_PATH = "results/model_comparison.csv"


# ==============================
# RESULTS
# ==============================

results = []


# ==============================
# EVALUATION LOOP
# ==============================

for model_name in MODELS:

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    # Model
    model = get_model(model_name).to(DEVICE)

    checkpoint_path = os.path.join(
        "checkpoints",
        f"{model_name}_best_model.pth"
    )

    # Special case: old R3D checkpoint
    if model_name == "r3d18":
        checkpoint_path = os.path.join(
            "checkpoints",
            "balanced_best_model.pth"
        )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE
        )
    )

    model.eval()

    # ==========================
    # CORRUPTION LOOP
    # ==========================

    for corruption in CORRUPTIONS:

        corruption_name = (
            "none"
            if corruption is None
            else corruption
        )

        print(
            f"\nTesting {model_name} | "
            f"Corruption: {corruption_name}"
        )

        test_dataset = ViolenceDataset(
            "test",
            corruption=corruption,
            corruption_probability=CORRUPTION_PROBABILITY
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        all_predictions = []
        all_labels = []

        with torch.no_grad():

            for videos, labels in test_loader:

                videos = videos.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(videos)

                predictions = outputs.argmax(dim=1)

                all_predictions.extend(
                    predictions.cpu().numpy()
                )

                all_labels.extend(
                    labels.cpu().numpy()
                )

        # ==========================
        # METRICS
        # ==========================

        accuracy = accuracy_score(
            all_labels,
            all_predictions
        )

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

        print(
            f"Accuracy: {accuracy:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

        # ==========================
        # SAVE RESULT
        # ==========================

        results.append({
            "model": model_name,
            "corruption": corruption_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })


# ==============================
# SAVE CSV
# ==============================

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print("\n" + "=" * 60)
print("ALL EVALUATIONS COMPLETE")
print("=" * 60)

print(
    f"\nResults saved to: {RESULTS_PATH}"
)

print("\nComparison:")
print(results_df.to_string(index=False))