from dataset import ViolenceDataset


# DATASET_PATH = "datasets/RLVS/Real Life Violence Dataset"

# dataset = ViolenceDataset(DATASET_PATH)

# print(f"Total samples: {len(dataset)}")
# print(f"First sample: {dataset.samples[0]}")
# print(f"Last sample: {dataset.samples[-1]}")

# video, label = dataset[0]

# print(f"Video tensor shape: {video.shape}")
# print(f"Label: {label}")
# print(f"Data type: {video.dtype}")
# print(f"Minimum value: {video.min()}")
# print(f"Maximum value: {video.max()}")



train_dataset = ViolenceDataset("train", corruption="mixed")
val_dataset = ViolenceDataset("val")
test_dataset = ViolenceDataset("test", corruption="mixed")


print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")


video, label = train_dataset[0]

print(f"\nVideo tensor shape: {video.shape}")
print(f"Label: {label}")
print(f"Data type: {video.dtype}")
print(f"Minimum value: {video.min()}")
print(f"Maximum value: {video.max()}")