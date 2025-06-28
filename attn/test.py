import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

ext = load(
    name="softmax",
    sources=["attn/main.cpp", "attn/flash_attention.cu"],
    extra_cuda_cflags=["-O2"],
    extra_cflags=["-O2"],
)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
q = torch.randn((2, 2, 4, 16)).to(DEVICE)
k = torch.randn_like(q)
v = torch.randn_like(q)

res = ext.forward(q, k, v)
print(f"res: {res}")
