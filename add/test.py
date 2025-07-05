import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

ext = load(
    name="add",
    sources=["./add/main.cpp", "./add/relu.cu"],
    extra_cuda_cflags=["-O2"],
    extra_cflags=["-O2"],
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dummy = torch.randn((32, 512, 256, 128), device=DEVICE)
NUM_ITERS = 1

# Create CUDA events
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# ==============================
# Benchmark PyTorch ReLU
# ==============================
start.record()
for _ in range(NUM_ITERS):
    res = F.relu(dummy, False)
end.record()

# Wait for everything to finish
torch.cuda.synchronize()
# Compute elapsed time (ms)
elapsed_ms = start.elapsed_time(end)
print(f"PyTorch: {elapsed_ms / 1000.0:.6f} seconds")

# ==============================
# Benchmark your CUDA ReLU
# ==============================
start.record()
for _ in range(NUM_ITERS):
    res1 = ext.relu(dummy, False)
end.record()

torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
print(f"CUDA: {elapsed_ms / 1000.0:.6f} seconds")
