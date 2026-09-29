import pandas as pd
from pathlib import Path


# ==============================
# PATHS
# ==============================

INPUT_PATH = Path("results/model_comparison.csv")
OUTPUT_DIR = Path("results")

TABLE_PATH = OUTPUT_DIR / "final_model_comparison.csv"
ROBUSTNESS_PATH = OUTPUT_DIR / "robustness_summary.csv"


# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(INPUT_PATH)


model_names = {
    "r3d18": "R3D-18",
    "r2plus1d18": "R(2+1)D-18",
    "mc3_18": "MC3-18",
    "s3d": "S3D"
}

df["model_name"] = df["model"].map(model_names)


# ==============================
# 1. FINAL COMPARISON TABLE
# ==============================

comparison_table = df[
    [
        "model_name",
        "corruption",
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]
].copy()

comparison_table.rename(
    columns={
        "model_name": "Model",
        "corruption": "Corruption",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score"
    },
    inplace=True
)

for column in ["Accuracy", "Precision", "Recall", "F1 Score"]:
    comparison_table[column] *= 100

comparison_table.to_csv(
    TABLE_PATH,
    index=False
)


# ==============================
# 2. ROBUSTNESS SUMMARY
# ==============================

robustness_results = []

for model in df["model"].unique():

    model_df = df[df["model"] == model].copy()

    clean_row = model_df[
        model_df["corruption"] == "none"
    ].iloc[0]

    clean_accuracy = clean_row["accuracy"]
    clean_f1 = clean_row["f1_score"]

    corrupted_df = model_df[
        model_df["corruption"] != "none"
    ]

    mean_accuracy = corrupted_df["accuracy"].mean()
    mean_f1 = corrupted_df["f1_score"].mean()

    accuracy_drop = (
        clean_accuracy - mean_accuracy
    ) * 100

    f1_drop = (
        clean_f1 - mean_f1
    ) * 100

    robustness_results.append({
        "Model": model_names[model],
        "Clean Accuracy (%)": clean_accuracy * 100,
        "Mean Corrupted Accuracy (%)": mean_accuracy * 100,
        "Accuracy Drop (%)": accuracy_drop,
        "Clean F1 (%)": clean_f1 * 100,
        "Mean Corrupted F1 (%)": mean_f1 * 100,
        "F1 Drop (%)": f1_drop
    })


robustness_df = pd.DataFrame(
    robustness_results
)

robustness_df.to_csv(
    ROBUSTNESS_PATH,
    index=False
)


# ==============================
# 3. PRINT RESULTS
# ==============================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    comparison_table.to_string(index=False)
)


print("\n" + "=" * 70)
print("ROBUSTNESS SUMMARY")
print("=" * 70)

print(
    robustness_df.to_string(index=False)
)


print("\nFiles generated:")
print(f"  {TABLE_PATH}")
print(f"  {ROBUSTNESS_PATH}")