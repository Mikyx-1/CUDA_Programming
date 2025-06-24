#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>



/**
 * @brief CUDA kernel to apply ReLU activation element-wise.
 * 
 * @param input Pointer to input tensor
 * @param output Pointer to output tensor
 * @param N     Total number of elements
 * 
*/
__global__
void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N)
    {
        output[tid] = fmaxf(0.0f, input[tid]);
    }
}



/**
 * @brief Applies ReLU activation, optionally in-place.
 * 
 * @param input A tensor of any shape.
 * @param in_place If true, modifies input directly.
 * 
 * @return ReLU-activated tensor (same shape as input).
 * 
*/
torch::Tensor relu(torch::Tensor input, bool in_place)
{

    int N = input.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    if (in_place == true)
    {
        relu_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            input.data_ptr<float>(),
            N
        );

        return input;
    }
    else
    {
        torch::Tensor output = torch::empty_like(input);
        relu_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            N
        );

        return output;
    }
}