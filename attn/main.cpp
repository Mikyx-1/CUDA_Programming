#include <torch/extension.h>

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", torch::wrap_pybind_function(forward), "Calculate self-attention of Q, K, V using flash attention algorithm");
}