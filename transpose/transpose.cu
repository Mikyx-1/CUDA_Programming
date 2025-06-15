#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void transpose_kernel(const float* input, float* output, int height, int width)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < height && x < width)
    {
        output[x*height + y] = input[y * width + x];
    }
}


torch::Tensor transpose(torch::Tensor input)
{
    int height = input.size(0);
    int width = input.size(1);

    auto output = torch::empty({width, height}, input.options());

    dim3 threads(16, 16);
    dim3 blocks((height + threads.x - 1) / threads.x, 
                (width + threads.y - 1) / threads.y);

    transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        height, 
        width
    );

    return output;
}