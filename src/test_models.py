from model import get_model

models = [
    "r3d18",
    "r2plus1d18",
    "mc3_18",
    "s3d"
]

for name in models:

    print(f"\nTesting: {name}")

    model = get_model(name)

    print("Model created successfully.")