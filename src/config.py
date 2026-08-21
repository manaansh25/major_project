from pathlib import Path
import torch


# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Dataset
# -----------------------------
NUM_CLASSES = 2
NUM_FRAMES = 16
FRAME_SIZE = 112


# -----------------------------
# Training
# -----------------------------
BATCH_SIZE = 8
EPOCHS = 5

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


# -----------------------------
# Checkpoints
# -----------------------------
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_model.pth"
ROBUST_BEST_MODEL_PATH = CHECKPOINT_DIR / "robust_best_model.pth"
ROBUST_LAST_MODEL_PATH = CHECKPOINT_DIR / "robust_last_model.pth"
BALANCED_BEST_MODEL_PATH = CHECKPOINT_DIR / "balanced_best_model.pth"
BALANCED_LAST_MODEL_PATH = CHECKPOINT_DIR / "balanced_last_model.pth"