#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define TILE_SIZE 8


__global__
void tiled_transpose_kernel(const float* input, float* output, int height, int width)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int y = blockIdx.x * blockDim.x + threadIdx.x; 
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < height && x < width)
    {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    __syncthreads();


    if (y < height && x < width)
    {
        output[x * height + y] = tile[threadIdx.y][threadIdx.x];
    }
}





torch::Tensor tiled_transpose(torch::Tensor input)
{
    int height = input.size(0);
    int width = input.size(1);

    auto output = torch::empty({width, height}, input.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((height + TILE_SIZE - 1) / TILE_SIZE, 
                (width + TILE_SIZE - 1) / TILE_SIZE);

    tiled_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        height,
        width
    );

    return output;
}