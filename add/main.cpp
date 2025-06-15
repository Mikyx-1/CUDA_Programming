#include <torch/extension.h>

torch::Tensor add(torch::Tensor a, torch::Tensor b);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add", torch::wrap_pybind_function(add), "Add 2 tensors");
}