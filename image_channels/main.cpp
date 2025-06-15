#include <torch/extension.h>


torch::Tensor to_grayscale(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("to_grayscale", torch::wrap_pybind_function(to_grayscale), "Convert the image to gray scale.");
}