from ultralytics import YOLO

# Model load
model = YOLO("yolo26m.pt")

# Laptop webcam par inference
model.predict(
    source=0,      # 0 = default webcam
    show=True,     # live window dikhayega
    conf=0.25      # confidence threshold
)