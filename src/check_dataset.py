from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


DATASET_PATH = Path("datasets/RLVS/Real Life Violence Dataset")
OUTPUT_PATH = Path("results/dataset_info.csv")

CLASSES = ["NonViolence", "Violence"]
VIDEO_EXTENSIONS = {".mp4", ".avi"}


def get_video_info(video_path, class_name):
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        return {
            "file_name": video_path.name,
            "class": class_name,
            "extension": video_path.suffix.lower(),
            "readable": False,
            "width": None,
            "height": None,
            "fps": None,
            "frame_count": None,
            "duration": None,
        }

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count / fps if fps > 0 else 0

    video.release()

    return {
        "file_name": video_path.name,
        "class": class_name,
        "extension": video_path.suffix.lower(),
        "readable": True,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration": round(duration, 2),
    }


def main():
    video_data = []

    for class_name in CLASSES:
        class_path = DATASET_PATH / class_name

        videos = [
            file
            for file in class_path.iterdir()
            if file.suffix.lower() in VIDEO_EXTENSIONS
        ]

        print(f"\nChecking {class_name}: {len(videos)} videos")

        for video_path in tqdm(videos, desc=class_name):
            info = get_video_info(video_path, class_name)
            video_data.append(info)

    dataframe = pd.DataFrame(video_data)

    dataframe.to_csv(OUTPUT_PATH, index=False)

    print("\nDataset check complete.")
    print(f"Total videos: {len(dataframe)}")
    print(f"Readable videos: {dataframe['readable'].sum()}")
    print(f"Unreadable videos: {(~dataframe['readable']).sum()}")
    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()