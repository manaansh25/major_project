import time
import torch

from config import DEVICE, BALANCED_BEST_MODEL_PATH
from torchvision.models.video import r3d_18

model = r3d_18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    2
)


# -----------------------------
# Configuration
# -----------------------------

WARMUP_BATCHES = 10
BENCHMARK_BATCHES = 50
BATCH_SIZE = 8


# -----------------------------
# Load model
# -----------------------------

model.load_state_dict(
    torch.load(
        BALANCED_BEST_MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# -----------------------------
# Create dummy input
# [B, C, T, H, W]
# -----------------------------

dummy_input = torch.randn(
    BATCH_SIZE,
    3,
    16,
    112,
    112,
    device=DEVICE
)


# -----------------------------
# Warm-up
# -----------------------------

print("Warming up GPU...")

with torch.no_grad():

    for _ in range(WARMUP_BATCHES):
        _ = model(dummy_input)

if DEVICE.type == "cuda":
    torch.cuda.synchronize()


# -----------------------------
# Benchmark
# -----------------------------

print("Running benchmark...")

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

start_time = time.perf_counter()

with torch.no_grad():

    for _ in range(BENCHMARK_BATCHES):
        _ = model(dummy_input)

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

end_time = time.perf_counter()


# -----------------------------
# Results
# -----------------------------

total_time = end_time - start_time

total_samples = BENCHMARK_BATCHES * BATCH_SIZE

fps = total_samples / total_time

latency_per_batch = total_time / BENCHMARK_BATCHES

latency_per_sample = latency_per_batch / BATCH_SIZE


print("\n--- INFERENCE BENCHMARK ---")

print(f"Device              : {DEVICE}")

print(f"Batch size           : {BATCH_SIZE}")

print(f"Input shape          : {tuple(dummy_input.shape)}")

print(f"Total samples        : {total_samples}")

print(f"Total inference time : {total_time:.4f} sec")

print(f"Latency / batch      : {latency_per_batch * 1000:.2f} ms")

print(f"Latency / sample     : {latency_per_sample * 1000:.2f} ms")

print(f"Throughput           : {fps:.2f} samples/sec")