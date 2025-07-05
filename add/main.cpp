#include <torch/extension.h>

torch::Tensor relu(torch::Tensor input, bool in_place);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("relu", torch::wrap_pybind_function(relu), "ReLU Activation");
}