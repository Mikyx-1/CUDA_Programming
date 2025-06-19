#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>



/**
 * @brief CUDA kernel to apply Leaky-ReLU activation element-wise.
 * 
 * @param input Pointer to input tensor
 * @param output Pointer to output tensor
 * @param N     Total number of elements
 * 
*/
__global__
void leaky_relu_kernel(const float* input, float* output, float negative_slope, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N)
    {
        output[tid] = fmaxf(input[tid], 0.0f) + negative_slope * fminf(0.0f, input[tid]);
    }
}



/**
 * @brief Applies Leaky-ReLU activation, optionally in-place.
 * 
 * @param input A tensor of any shape.
 * @param in_place If true, modifies input directly.
 * 
 * @return LeakyReLU-activated tensor (same shape as input).
 * 
*/
torch::Tensor leaky_relu(torch::Tensor input, float negative_slope = 0.01f, bool in_place = false)
{
    int N = input.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    if (in_place == true)
    {
        leaky_relu_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            input.data_ptr<float>(),
            negative_slope,
            N
        );
    }
    else{
        torch::Tensor output = torch::empty_like(input);
        leaky_relu_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            negative_slope,
            N
        );
    }
}