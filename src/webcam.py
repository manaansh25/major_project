import cv2
import torch
import numpy as np
from collections import deque

from model import get_model
from config import DEVICE


# ==============================
# SETTINGS
# ==============================

MODEL_NAME = "r3d18"

CHECKPOINT_PATH = "checkpoints/balanced_best_model.pth"

NUM_FRAMES = 16
FRAME_SIZE = 112

MEAN = [0.43216, 0.394666, 0.37645]
STD = [0.22803, 0.22145, 0.216989]

CLASS_NAMES = [
    "Non-Violence",
    "Violence"
]


# ==============================
# MODEL
# ==============================

print(f"Loading {MODEL_NAME}...")

model = get_model(MODEL_NAME).to(DEVICE)

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully.")


# ==============================
# WEBCAM
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

frame_buffer = deque(maxlen=NUM_FRAMES)


# ==============================
# PREPROCESS
# ==============================

def preprocess_frame(frame):

    # BGR -> RGB
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Same resize as dataset.py
    frame = cv2.resize(
        frame,
        (FRAME_SIZE, FRAME_SIZE)
    )

    # Same normalization
    frame = frame.astype(
        np.float32
    ) / 255.0

    return frame


def preprocess_clip(frames):

    frames_array = np.stack(frames)

    frames_tensor = torch.from_numpy(
        frames_array
    )

    # T H W C -> C T H W
    frames_tensor = frames_tensor.permute(
        3, 0, 1, 2
    )

    mean = torch.tensor(
        MEAN,
        dtype=torch.float32
    ).view(3, 1, 1, 1)

    std = torch.tensor(
        STD,
        dtype=torch.float32
    ).view(3, 1, 1, 1)

    frames_tensor = (
        frames_tensor - mean
    ) / std

    # Add batch dimension
    frames_tensor = frames_tensor.unsqueeze(0)

    return frames_tensor


# ==============================
# WEBCAM LOOP
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    processed_frame = preprocess_frame(frame)

    frame_buffer.append(processed_frame)


    # ==========================
    # PREDICTION
    # ==========================

    if len(frame_buffer) == NUM_FRAMES:

        video = preprocess_clip(
            list(frame_buffer)
        ).to(DEVICE)

        with torch.no_grad():

            outputs = model(video)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            prediction = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = probabilities[
                0,
                prediction
            ].item()

        label = CLASS_NAMES[prediction]

        # ==========================
        # DISPLAY
        # ==========================

        text = (
            f"{label}: "
            f"{confidence * 100:.1f}%"
        )

        if prediction == 1:
            text_color = (0, 0, 255)
        else:
            text_color = (0, 255, 0)

        cv2.putText(
            frame,
            text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            text_color,
            2
        )

    else:

        cv2.putText(
            frame,
            f"Collecting frames: "
            f"{len(frame_buffer)}/{NUM_FRAMES}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    cv2.imshow(
        "Violence Detection",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()