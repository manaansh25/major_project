from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_PATH = Path("datasets/RLVS/Real Life Violence Dataset")
OUTPUT_PATH = Path("results/splits.csv")

CLASS_TO_LABEL = {
    "NonViolence": 0,
    "Violence": 1,
}

VIDEO_EXTENSIONS = {".mp4", ".avi"}

RANDOM_SEED = 42


def collect_videos():
    samples = []

    for class_name, label in CLASS_TO_LABEL.items():
        class_path = DATASET_PATH / class_name

        for video_path in class_path.iterdir():
            if video_path.suffix.lower() in VIDEO_EXTENSIONS:
                samples.append({
                    "video_path": str(video_path),
                    "label": label,
                    "class": class_name,
                })

    return pd.DataFrame(samples)


def main():
    df = collect_videos()

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    splits_df = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True,
    )

    splits_df.to_csv(OUTPUT_PATH, index=False)

    print("\n--- SPLIT SUMMARY ---")
    print(splits_df.groupby(["split", "class"]).size())

    print(f"\nTotal samples: {len(splits_df)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()