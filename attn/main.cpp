#include <torch/extension.h>

torch::Tensor self_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("self_attention", torch::wrap_pybind_function(self_attention), "Calculate self-attention of Q, K, V");
}