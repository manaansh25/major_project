import subprocess
import sys

MODELS = [
    "mc3_18",
    "s3d"
]

for model_name in MODELS:

    print("\n" + "=" * 70)
    print(f"STARTING TRAINING: {model_name}")
    print("=" * 70)

    result = subprocess.run(
        [
            sys.executable,
            "src/train.py",
            "--model",
            model_name
        ]
    )

    if result.returncode != 0:
        print(f"\nTraining FAILED for {model_name}.")
        print("Stopping remaining models.")
        sys.exit(result.returncode)

    print(f"\n{model_name} training completed successfully.")

print("\n" + "=" * 70)
print("ALL REMAINING MODELS TRAINED SUCCESSFULLY")
print("=" * 70)