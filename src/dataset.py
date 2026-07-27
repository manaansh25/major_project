from pathlib import Path

import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


NUM_FRAMES = 16
FRAME_SIZE = 112

VIDEO_EXTENSIONS = {".mp4", ".avi"}

CLASS_TO_LABEL = {
    "NonViolence": 0,
    "Violence": 1,
}


class ViolenceDataset(Dataset):

    #first we were scanning the folders and loading all the videos
    # def __init__(self, dataset_path):
    #     self.dataset_path = Path(dataset_path)
    #     self.samples = []

    #     for class_name, label in CLASS_TO_LABEL.items():
    #         class_path = self.dataset_path / class_name

    #         for video_path in class_path.iterdir():
    #             if video_path.suffix.lower() in VIDEO_EXTENSIONS:
    #                 self.samples.append((video_path, label))
   
   #scan split-wise as per the requirement
    def __init__(self, split, splits_path="results/splits.csv"):
        self.split = split
        self.splits_path = Path(splits_path)

        splits_df = pd.read_csv(self.splits_path)

        split_df = splits_df[splits_df["split"] == split]

        self.samples = [
            (Path(row["video_path"]), int(row["label"]))
            for _, row in split_df.iterrows()
        ]

    def __len__(self):
        return len(self.samples)
    
    # #loading videos into RAM and then reading 
    # def _read_video(self, video_path):
    #     video = cv2.VideoCapture(str(video_path))
    #     frames = []

    #     while True:
    #         success, frame = video.read()

    #         if not success:
    #             break

    #         frames.append(frame)

    #     video.release()

    #     return frames
    
    # #dividing the video into frames and sampling 16 frames from the video
    # def _sample_frames(self, frames):
    #     total_frames = len(frames)

    #     indices = np.linspace(
    #         0,
    #         total_frames - 1,
    #         NUM_FRAMES,
    #         dtype=int
    #     )

    #     sampled_frames = [frames[index] for index in indices]
    #     return sampled_frames
    
    #improvement: reading frames directly from the video file without loading the entire video into RAM
    def _load_frames(self, video_path):
        video = cv2.VideoCapture(str(video_path))

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        indices = np.linspace(
            0,
            total_frames - 1,
            NUM_FRAMES,
            dtype=int
        )

        frames = []

        for index in indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, index)

            success, frame = video.read()

            if success:
                frames.append(frame)

        video.release()

        #corrupt frame reading safety check
        if len(frames) == 0:
            raise RuntimeError(f"Could not read frames from: {video_path}")

        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1].copy())
        
        return frames    

    
    def _preprocess_frames(self, frames):
        processed_frames = []

        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))

            frame = frame.astype(np.float32) / 255.0

            processed_frames.append(frame)

        frames_array = np.stack(processed_frames)

        frames_tensor = torch.from_numpy(frames_array)

        frames_tensor = frames_tensor.permute(3, 0, 1, 2)

        return frames_tensor
    
    
    def __getitem__(self, index):
        video_path, label = self.samples[index]

        frames = self._load_frames(video_path)
        frames_tensor = self._preprocess_frames(frames)

        return frames_tensor, label