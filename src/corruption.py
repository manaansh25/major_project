import cv2
import numpy as np

def add_gaussian_noise(frames, mean=0, sigma=15):

    noisy_frames = []

    for frame in frames:

        noise = np.random.normal(
            mean,
            sigma,
            frame.shape
        ).astype(np.float32)

        noisy = frame.astype(np.float32) + noise

        noisy = np.clip(noisy, 0, 255)

        noisy_frames.append(
            noisy.astype(np.uint8)
        )

    return noisy_frames

def add_motion_blur(frames, kernel_size=9):

    blurred_frames = []

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    kernel[kernel_size // 2, :] = np.ones(kernel_size)

    kernel = kernel / kernel_size

    for frame in frames:

        blurred = cv2.filter2D(frame, -1, kernel)

        blurred_frames.append(blurred)

    return blurred_frames

def reduce_brightness(frames, factor=0.5):

    dark_frames = []

    for frame in frames:

        dark = frame.astype(np.float32) * factor

        dark = np.clip(dark, 0, 255)

        dark_frames.append(
            dark.astype(np.uint8)
        )

    return dark_frames

def add_gaussian_blur(frames, kernel_size=5):

    blurred_frames = []

    for frame in frames:

        blurred = cv2.GaussianBlur(
            frame,
            (kernel_size, kernel_size),
            0
        )

        blurred_frames.append(blurred)

    return blurred_frames

def add_mixed_corruption(frames):

    frames = add_gaussian_noise(
        frames,
        sigma=30
    )

    frames = add_motion_blur(
        frames,
        kernel_size=9
    )

    frames = reduce_brightness(
        frames,
        factor=0.6
    )

    return frames