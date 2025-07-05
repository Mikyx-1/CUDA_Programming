#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief Inline device function for ReLU activation on single float
 */
__forceinline__ __device__ float relu_activation(float x)
{
    return fmaxf(0.0f, x);
}

/**
 * @brief Inline device function for ReLU activation on float2
 */
__forceinline__ __device__ float2 relu_activation_float2(float2 val)
{
    val.x = relu_activation(val.x);
    val.y = relu_activation(val.y);
    return val;
}

/**
 * @brief CUDA kernel to apply ReLU activation using float2 (2 elements per thread).
 * 
 * @param input Pointer to input tensor
 * @param output Pointer to output tensor
 * @param N     Total number of elements
 */
__global__
void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int vecN = N >> 1;  // N / 2 using bit shift

    // Process 2 elements at a time
    if (tid < vecN)
    {
        float2 val = reinterpret_cast<const float2*>(input)[tid];
        reinterpret_cast<float2*>(output)[tid] = relu_activation_float2(val);
    }

    // Handle last element if N is odd
    if (tid == 0 && (N & 1))  // N % 2 != 0 using bit mask
    {
        output[N - 1] = relu_activation(input[N - 1]);
    }
}

/**
 * @brief Applies ReLU activation, optionally in-place.
 * 
 * @param input A tensor of any shape.
 * @param in_place If true, modifies input directly.
 * 
 * @return ReLU-activated tensor (same shape as input).
 */
torch::Tensor relu(torch::Tensor input, bool in_place)
{
    int N = input.numel();
    int threads = 256;
    int blocks = ((N >> 1) + threads - 1) / threads;  // Calculate blocks for float2 processing

    if (in_place)
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