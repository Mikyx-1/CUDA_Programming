import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import math
import time

ext = load(
    name="softmax",
    sources=["attn/main.cpp", "attn/flash_attention.cu"],
    extra_cuda_cflags=["-O2"],
    extra_cflags=["-O2"],
)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
q = torch.randn((2, 16, 48, 13)).to(DEVICE)  # Currently, must be divisible by 32
k = torch.randn_like(q)
v = torch.randn_like(q)

def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


tic = time.time()
for _ in range(1000):
    res = ext.flash_attention(q, k, v)
toc = time.time()
print(f"Duration: {toc - tic}")

tic = time.time()
for _ in range(1000):
    gt = manual_attn(q, k, v)
toc = time.time()
print(f"Duration: {toc - tic}")


print(f"all_close: {torch.allclose(res, gt)}, Max Diff: {torch.abs(res - gt).max()}")