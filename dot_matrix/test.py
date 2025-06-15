import torch
from torch.utils.cpp_extension import load

ext = load(
    name="dot_matrix",
    sources=["dot_matrix/main.cpp", "dot_matrix/dot_matrix.cu"],
    extra_cuda_cflags=["-O2"],
    extra_cflags=["-O2"],
)

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
a = torch.randn((2, 3)).to(DEVICE)
b = torch.randn((3, 2)).to(DEVICE)
gt = torch.matmul(a, b)
res = ext.dot_matrix(a, b)
print(f"res: {res}")
print(f"gt: {gt}")
print(f"all_close: {torch.allclose(res, gt)}")
