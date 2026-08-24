import numpy as np
import matplotlib.pyplot as plt

results = {
    "Clean": [[140, 10], [1, 149]],
    "Gaussian Noise": [[139, 11], [0, 150]],
    "Motion Blur": [[136, 14], [1, 149]],
    "Brightness Reduction": [[145, 5], [4, 146]],
    "Gaussian Blur": [[141, 9], [1, 149]],
    "Mixed": [[144, 6], [5, 145]]
}

for condition, cm in results.items():

    cm = np.array(cm)

    plt.figure(figsize=(6, 5))

    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {condition}")
    plt.colorbar()

    classes = ["Non-Violence", "Violence"]

    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    filename = condition.lower().replace(" ", "_")
    plt.savefig(
        f"results/confusion_matrix_{filename}.png",
        dpi=300
    )

    plt.close()

print("All confusion matrices saved.")