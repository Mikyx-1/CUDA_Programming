#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__
void softplus_kernel(const float* input, float* output, int N, float beta = 1.0f, float threshold = 20.0f)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {   
        if (input[idx] * beta <= threshold)
        {
            output[idx] = (1 / beta) * __logf(1.0f + __expf(beta * input[idx]));
        }

        else{
            // To avoid numerical overflow
            output[idx] = input[idx];
        }

    }
}


torch::Tensor softplus(torch::Tensor input, bool in_place = false, float beta = 1.0f, float threshold = 20.0f)
{

    int N = input.numel();

    int threads = 256; 
    int blocks = (N + threads - 1) / threads;

    if (in_place)
    {
        softplus_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            input.data_ptr<float>(),
            N,
            beta,
            threshold
        );

        return input;
    }
    else{

        torch::Tensor output = torch::empty_like(input);
        softplus_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            N,
            beta,
            threshold
        );

        return output;
    }
}