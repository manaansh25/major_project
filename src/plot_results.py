import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------

RESULTS_FILE = Path("results/final_results.csv")
OUTPUT_DIR = Path("results/plots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load results
# -----------------------------

df = pd.read_csv(RESULTS_FILE)

# -----------------------------
# Plot 1: Accuracy
# -----------------------------

accuracy_df = df.pivot(
    index="Condition",
    columns="Model",
    values="Accuracy"
)

accuracy_df.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Test Condition")
plt.title("Accuracy Under Different Visual Degradations")
plt.xticks(rotation=25)
plt.legend(title="Model")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "accuracy_comparison.png",
    dpi=300
)

plt.close()

# -----------------------------
# Plot 2: F1 Score
# -----------------------------

f1_df = df.pivot(
    index="Condition",
    columns="Model",
    values="F1"
)

f1_df.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.ylabel("F1 Score (%)")
plt.xlabel("Test Condition")
plt.title("F1 Score Under Different Visual Degradations")
plt.xticks(rotation=25)
plt.legend(title="Model")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "f1_comparison.png",
    dpi=300
)

plt.close()

# -----------------------------
# Plot 3: Clean vs Mixed
# -----------------------------

selected = df[
    df["Condition"].isin(["Clean", "Mixed"])
]

clean_mixed = selected.pivot(
    index="Model",
    columns="Condition",
    values="Accuracy"
)

clean_mixed.plot(
    kind="bar",
    figsize=(9, 6)
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("Clean vs Mixed-Degradation Performance")
plt.xticks(rotation=15)
plt.legend(title="Condition")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "clean_vs_mixed.png",
    dpi=300
)

plt.close()

print("Plots generated successfully.")

print("\nSaved files:")

for file in OUTPUT_DIR.iterdir():
    print(file)