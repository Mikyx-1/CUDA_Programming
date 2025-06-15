import torch
from torch.utils.cpp_extension import load

ext = load(
    name="softmax",
    sources=["softmax/main.cpp", "softmax/softmax.cu"],
    extra_cuda_cflags=["-O2"],
    extra_cflags=["-O2"],
)

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
a = torch.randn((2000, 3)).to(DEVICE)

res = ext.softmax(a)
gt = torch.softmax(a, -1)
print(f"res: {res}")
print(f"gt: {gt}")
print(f"all_close: {torch.allclose(res, gt)}")
