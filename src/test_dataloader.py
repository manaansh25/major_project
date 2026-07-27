from dataloader import train_loader

videos, labels = next(iter(train_loader))

print(f"Videos shape : {videos.shape}")
print(f"Labels shape : {labels.shape}")

print(f"Videos dtype : {videos.dtype}")
print(f"Labels dtype : {labels.dtype}")

print(f"Min value : {videos.min()}")
print(f"Max value : {videos.max()}")

print(f"Labels : {labels}")