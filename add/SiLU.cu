#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void silu_kernel(const float* input, float* output, int N)
{

    int idx = blockIdx.x  * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        output[idx] = input[idx] / (1.0f + __expf(-input[idx]));
    }
}



torch::Tensor SiLU(torch::Tensor input, bool in_place = false)
{
    int N = input.numel();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    if (in_place)
    {
        silu_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            input.data_ptr<float>(),
            N
        );

        return input;
    }
    else{
        torch::Tensor output = torch::empty_like(input);
        silu_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            N
        );

        return output;
    }
}