#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void elu_kernel(const float* input, float* output, float alpha, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        float x = input[idx];
        output[idx] = x > 0 ? x : alpha * (__expf(x) - 1.0f);
    }
}

torch::Tensor ELU(torch::Tensor input, float alpha = 1.0f, bool in_place = false)
{
    int N = input.numel();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    if (in_place)
    {
        elu_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            input.data_ptr<float>(),
            alpha,
            N
        );

        return input;
    }
    else
    {
        auto output = torch::empty_like(input);
        elu_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            alpha,
            N
        );

        return output;
    }
}
