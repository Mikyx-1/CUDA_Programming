#include <torch/extension.h>

torch::Tensor transpose(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("transpose", torch::wrap_pybind_function(transpose), "Transpose a 2D matrix");
}