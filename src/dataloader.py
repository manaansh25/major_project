from torch.utils.data import DataLoader
from dataset import ViolenceDataset


BATCH_SIZE = 8


train_dataset = ViolenceDataset("train")
val_dataset = ViolenceDataset("val")
test_dataset = ViolenceDataset("test")


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)