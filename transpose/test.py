import torch
from torch.utils.cpp_extension import load

ext = load(
    name="transpose",
    sources=["./transpose/main.cpp", "./transpose/transpose.cu"],
    extra_cuda_cflags=["-O2"],
    extra_cflags=["-O2"],
)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a = torch.randn((3, 4)).to(DEVICE)
transposed = ext.transpose(a)
print(f"a: {a}")
print(f"transposed: {transposed}")
