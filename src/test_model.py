from model import get_model

model = get_model()

print(model)

print("\nFinal Layer:")
print(model.fc)