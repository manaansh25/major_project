import pandas as pd


DATA_PATH = "results/dataset_info.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("\n--- DATASET SUMMARY ---")
    print(f"Total videos: {len(df)}")

    print("\n--- CLASS DISTRIBUTION ---")
    print(df["class"].value_counts())

    print("\n--- FILE FORMATS ---")
    print(df["extension"].value_counts())

    print("\n--- VIDEO RESOLUTIONS ---")
    print(
        df.groupby(["width", "height"])
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\n--- FPS STATISTICS ---")
    print(df["fps"].describe())

    print("\n--- FRAME COUNT STATISTICS ---")
    print(df["frame_count"].describe())

    print("\n--- DURATION STATISTICS (SECONDS) ---")
    print(df["duration"].describe())

    print("\n--- STATISTICS BY CLASS ---")
    print(
        df.groupby("class")[
            ["fps", "frame_count", "duration"]
        ].mean()
    )


if __name__ == "__main__":
    main()