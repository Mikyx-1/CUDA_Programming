#include <torch/extension.h>

torch::Tensor dot_matrix(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dot_matrix", torch::wrap_pybind_function(dot_matrix), "dot 2 matrices");
}