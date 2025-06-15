import math

import torch
from torch.utils.cpp_extension import load

ext = load(
    name="add",
    sources=["./add/main.cpp", "./add/add_advanced.cu"],
    extra_cuda_cflags=["-O2"],
    extra_cflags=["-O2"],
)

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
a = torch.randn((1, 3, 6, 4)).to(DEVICE)
b = torch.randn((1, 3, 6, 4)).to(DEVICE)

res = ext.add(a, b)
res1 = a + b

print(f"res: {res}")
print(f"res1: {res1}")
print(f"all_close: {torch.allclose(res, res1, atol=1e-7)}")