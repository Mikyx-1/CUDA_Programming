#include <torch/extension.h>

torch::Tensor convol2D(torch::Tensor input, torch::Tensor kernel);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("convol2D", torch::wrap_pybind_function(convol2D), "2D Convolution.");
}