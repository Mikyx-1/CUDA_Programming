import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

ext = load(
    name="convol2d",
    sources=["convolution/main.cpp", "convolution/convolution.cu"],
    extra_cuda_cflags=["-O2"],
    extra_cflags=["-O2"],
)

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
matrix = torch.randn((512, 256)).to(DEVICE)
kernel = torch.randn((3, 3)).to(DEVICE)

res = ext.convol2D(matrix, kernel)
gt = F.conv2d(
    matrix.expand(1, 1, matrix.shape[0], matrix.shape[1]),
    kernel.expand(1, 1, kernel.shape[0], kernel.shape[1]),
    padding=((kernel.shape[0] - 1) // 2, (kernel.shape[1] - 1) // 2),
)
print(res)
print(gt)
print(
    f"all_close: {torch.allclose(res, gt, atol=1e-6)}, diff: {torch.linalg.norm(res - gt)}"
)
