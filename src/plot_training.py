import pandas as pd
import matplotlib.pyplot as plt

HISTORY_PATH = "results/training_history.csv"

df = pd.read_csv(HISTORY_PATH)

# Training vs Validation Loss
plt.figure(figsize=(8, 5))

plt.plot(
    df["epoch"],
    df["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    df["epoch"],
    df["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("results/training_loss.png", dpi=300)
plt.close()


# Training vs Validation Accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    df["epoch"],
    df["train_accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    df["epoch"],
    df["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("results/training_accuracy.png", dpi=300)
plt.close()

print("Training curves saved:")
print("results/training_loss.png")
print("results/training_accuracy.png")