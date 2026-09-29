import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from dataset import ViolenceDataset
from model import get_model
from config import DEVICE


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

OUTPUT_DIR = "results/confusion_matrices"

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
# MODEL NAMES
# ==============================

model_names = {
    "r3d18": "R3D-18",
    "r2plus1d18": "R(2+1)D-18",
    "mc3_18": "MC3-18",
    "s3d": "S3D"
}


# ==============================
# EVALUATION
# ==============================

for model_name in MODELS:

    print(f"\nProcessing {model_names[model_name]}...")

    model = get_model(model_name).to(DEVICE)

    checkpoint_path = os.path.join(
        "checkpoints",
        f"{model_name}_best_model.pth"
    )

    # Existing R3D checkpoint
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

    for corruption in CORRUPTIONS:

        corruption_name = (
            "none"
            if corruption is None
            else corruption
        )

        print(f"  {corruption_name}")

        dataset = ViolenceDataset(
            "test",
            corruption=corruption,
            corruption_probability=1.0
        )

        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        all_predictions = []
        all_labels = []

        with torch.no_grad():

            for videos, labels in loader:

                videos = videos.to(DEVICE)

                outputs = model(videos)

                predictions = outputs.argmax(dim=1)

                all_predictions.extend(
                    predictions.cpu().numpy()
                )

                all_labels.extend(
                    labels.numpy()
                )

        cm = confusion_matrix(
            all_labels,
            all_predictions
        )

        # ==========================
        # PLOT
        # ==========================

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Non-Violence",
                "Violence"
            ]
        )

        display.plot(
            ax=ax,
            values_format="d"
        )

        ax.set_title(
            f"{model_names[model_name]} - "
            f"{corruption_name.replace('_', ' ').title()}"
        )

        plt.tight_layout()

        filename = (
            f"{model_name}_{corruption_name}.png"
        )

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                filename
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


print("\nAll confusion matrices generated.")

print(f"Saved to: {OUTPUT_DIR}")