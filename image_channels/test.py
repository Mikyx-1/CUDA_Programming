import torch
from torch.utils.cpp_extension import load

ext = load(
    name="to_grayscale",
    sources=["./image_channels/main.cpp", "./image_channels/to_grayscale.cu"],
    extra_cuda_cflags=["-O2"],
    extra_cflags=["-O2"],
)

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
dummy = torch.ones((1, 3, 2, 2)).to(DEVICE)

res = ext.to_grayscale(dummy)
gt = torch.full(
    (dummy.shape[0], 1, dummy.shape[2], dummy.shape[3]),
    -1,
    device=dummy.device,
    dtype=dummy.dtype,
)
print(f"res: {res}")
print(f"gt: {gt}")
print(f"all_close: {torch.allclose(res, gt)}")
