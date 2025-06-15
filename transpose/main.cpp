#include <torch/extension.h>

torch::Tensor tiled_transpose(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tiled_transpose", torch::wrap_pybind_function(tiled_transpose), "Transpose a 2D matrix");
}