import torch
from torch.utils.cpp_extension import load

ext = load(
    name="transpose",
    sources=["./transpose/main.cpp", "./transpose/tiled_transpose.cu"],
    extra_cuda_cflags=["-O2"],
    extra_cflags=["-O2"],
)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a = torch.randn((2048, 8192)).to(DEVICE)
transposed = ext.tiled_transpose(a)
transposed_torch = a.t()
print(f"a: {a}")
print(f"transposed: {transposed}")
print(f"gt: {transposed_torch}")
print(f"all_close: {torch.allclose(transposed, transposed_torch)}")