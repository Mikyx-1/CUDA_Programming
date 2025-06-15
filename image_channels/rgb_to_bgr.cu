#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void rgb_to_bgr_kernel(const float* input, float* output, int height, int width, int area)
{
    int batch_id = blockIdx.x;
    int y = threadIdx.y;
    int x = threadIdx.x;

    if (y < height && x < width)
    {
        int r_idx = batch_id * 3 * area + 0 * area + y * width + x;
        int g_idx = batch_id * 3 * area + 1 * area + y * width + x;
        int b_idx = batch_id * 3 * area + 2 * area + y * width + x;

        output[r_idx] = input[b_idx];
        output[g_idx] = input[g_idx];
        output[b_idx] = input[r_idx];

    }
}




torch::Tensor rgb_to_bgr(torch::Tensor input)
{

    int num_channels = input.size(1);
    TORCH_CHECK(num_channels == 3, "Input must have 3 channels (RGB)");
    
    int batch_size = input.size(0);
    int height = input.size(2);
    int width = input.size(3);
    int area = height * width;

    auto output = torch::empty_like(input);

    dim3 threads(16, 16);
    dim3 blocks(batch_size);

    rgb_to_bgr_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        height,
        width,
        area
    );

    return output;
}