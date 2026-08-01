import cv2
from pathlib import Path

from corruption import (
    add_gaussian_noise,
    add_motion_blur,
    reduce_brightness,
    add_gaussian_blur,
    add_mixed_corruption
)


video_path = Path(
    "datasets/RLVS/Real Life Violence Dataset/Violence/V_1.mp4"
)

cap = cv2.VideoCapture(str(video_path))

frames = []

while len(frames) < 16:

    success, frame = cap.read()

    if not success:
        break

    frames.append(frame)

cap.release()

# noisy_frames = add_gaussian_noise(
#     frames,
#     sigma=30
# )

# blurred_frames = add_motion_blur(
#     frames,
#     kernel_size=9
# )

# dark_frames = reduce_brightness(
#     frames,
#     factor=0.4
# )

# blurred_frames = add_gaussian_blur(
#     frames,
#     kernel_size=9
# )

mixed_frames = add_mixed_corruption(
    frames
)

cv2.imwrite(
    "results/original_frame.jpg",
    frames[0]
)

# cv2.imwrite(
#     "results/noisy_frame.jpg",
#     noisy_frames[0]
# )

# cv2.imwrite(
#     "results/motion_blur_frame.jpg",
#     blurred_frames[0]
# )

# cv2.imwrite(
#     "results/dark_frame.jpg",
#     dark_frames[0]
# )

# cv2.imwrite(
#     "results/gaussian_blur_frame.jpg",
#     blurred_frames[0]
# )

cv2.imwrite(
    "results/mixed_frame.jpg",
    mixed_frames[0]
)

print("Images saved successfully.")