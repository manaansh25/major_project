import pandas as pd
import matplotlib.pyplot as plt

RESULTS_PATH = "results/final_results.csv"

df = pd.read_csv(RESULTS_PATH)

models = [
    "Baseline R3D-18",
    "100% Mixed R3D-18",
    "50/50 Proposed R3D-18"
]

conditions = [
    "Clean",
    "Gaussian Noise",
    "Motion Blur",
    "Brightness Reduction",
    "Gaussian Blur",
    "Mixed"
]

df = df[
    df["Model"].isin(models) &
    df["Condition"].isin(conditions)
]

pivot = df.pivot(
    index="Condition",
    columns="Model",
    values="Accuracy"
)

pivot = pivot.reindex(conditions)
pivot = pivot[models]

plt.figure(figsize=(10, 6))

pivot.plot(
    kind="bar",
    figsize=(10, 6),
    width=0.8
)

plt.xlabel("Test Condition")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison Across Test Conditions")
plt.xticks(rotation=0)
plt.legend(title="Training Strategy")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/model_comparison_accuracy.png",
    dpi=300
)

plt.close()

print("Comparison plot saved to:")
print("results/model_comparison_accuracy.png")