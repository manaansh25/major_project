import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ==============================
# PATHS
# ==============================

INPUT_PATH = Path("results/model_comparison.csv")
OUTPUT_DIR = Path("results/model_plots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# LOAD RESULTS
# ==============================

df = pd.read_csv(INPUT_PATH)

corruption_order = [
    "none",
    "gaussian",
    "motion",
    "brightness",
    "gaussian_blur",
    "mixed"
]

model_order = [
    "r3d18",
    "r2plus1d18",
    "mc3_18",
    "s3d"
]

df["corruption"] = pd.Categorical(
    df["corruption"],
    categories=corruption_order,
    ordered=True
)

df["model"] = pd.Categorical(
    df["model"],
    categories=model_order,
    ordered=True
)

df = df.sort_values(
    ["model", "corruption"]
)


# ==============================
# DISPLAY NAMES
# ==============================

model_names = {
    "r3d18": "R3D-18",
    "r2plus1d18": "R(2+1)D-18",
    "mc3_18": "MC3-18",
    "s3d": "S3D"
}

corruption_names = {
    "none": "None",
    "gaussian": "Gaussian",
    "motion": "Motion",
    "brightness": "Brightness",
    "gaussian_blur": "Gaussian Blur",
    "mixed": "Mixed"
}


# ==============================
# 1. ACCURACY PLOT
# ==============================

plt.figure(figsize=(10, 6))

for model in model_order:

    subset = df[df["model"] == model]

    plt.plot(
        subset["corruption"],
        subset["accuracy"] * 100,
        marker="o",
        label=model_names[model]
    )

plt.xticks(
    range(len(corruption_order)),
    [corruption_names[x] for x in corruption_order],
    rotation=20
)

plt.xlabel("Corruption Type")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Under Different Visual Conditions")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "accuracy_comparison.png",
    dpi=300
)

plt.close()


# ==============================
# 2. F1 SCORE PLOT
# ==============================

plt.figure(figsize=(10, 6))

for model in model_order:

    subset = df[df["model"] == model]

    plt.plot(
        subset["corruption"],
        subset["f1_score"] * 100,
        marker="o",
        label=model_names[model]
    )

plt.xticks(
    range(len(corruption_order)),
    [corruption_names[x] for x in corruption_order],
    rotation=20
)

plt.xlabel("Corruption Type")
plt.ylabel("F1 Score (%)")
plt.title("Model F1 Score Under Different Visual Conditions")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "f1_comparison.png",
    dpi=300
)

plt.close()


# ==============================
# 3. CLEAN MODEL COMPARISON
# ==============================

clean_df = df[df["corruption"] == "none"]

plt.figure(figsize=(8, 6))

plt.bar(
    [model_names[x] for x in model_order],
    clean_df.set_index("model").loc[model_order]["accuracy"] * 100
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("Clean-Test Accuracy Comparison")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "clean_accuracy_comparison.png",
    dpi=300
)

plt.close()


# ==============================
# 4. ROBUSTNESS DROP
# ==============================

robustness_rows = []

for model in model_order:

    model_df = df[df["model"] == model]

    clean_accuracy = model_df[
        model_df["corruption"] == "none"
    ]["accuracy"].iloc[0]

    for _, row in model_df.iterrows():

        if row["corruption"] == "none":
            continue

        drop = (
            clean_accuracy - row["accuracy"]
        ) * 100

        robustness_rows.append({
            "model": model_names[model],
            "corruption": corruption_names[row["corruption"]],
            "accuracy_drop": drop
        })


robustness_df = pd.DataFrame(
    robustness_rows
)

plt.figure(figsize=(10, 6))

for model in model_names.values():

    subset = robustness_df[
        robustness_df["model"] == model
    ]

    plt.plot(
        subset["corruption"],
        subset["accuracy_drop"],
        marker="o",
        label=model
    )

plt.axhline(
    0,
    linewidth=1
)

plt.xticks(
    rotation=20
)

plt.xlabel("Corruption Type")
plt.ylabel("Accuracy Drop from Clean (%)")
plt.title("Robustness: Accuracy Drop Under Corruption")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "robustness_drop.png",
    dpi=300
)

plt.close()


print("\nPlots generated successfully.")

print(f"\nSaved to: {OUTPUT_DIR}")

print("\nFiles:")
print("1. accuracy_comparison.png")
print("2. f1_comparison.png")
print("3. clean_accuracy_comparison.png")
print("4. robustness_drop.png")