#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__
void to_grayscale_kernel(const float* input, float* output, int height, int width)
{
    // Temporarily set them to 0
    int batch_id = blockIdx.x;
    int y = threadIdx.y;
    int x = threadIdx.x;

    if (y < height && x < width)
    {
        float r = input[(batch_id * 3 * height * width) + 0 * y * width + x];
        float g = input[(batch_id * 3 * height * width) + 1 * y * width + x];
        float b = input[(batch_id * 3 * height * width) + 2 * y * width + x];
        output[batch_id * height * width + y * width + x] = 0.299 * r + 0.587 * g + 0.114 * b;
    }

}


torch::Tensor to_grayscale(torch::Tensor input)
{

    int batch_size = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    TORCH_CHECK(channels==3, "Input must have 3 channels (RGB)");

    auto output = torch::empty({batch_size, 1, height, width}, input.options());

    dim3 threads(16, 16);
    dim3 blocks(batch_size);

    to_grayscale_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        height,
        width
    );

    return output;
}