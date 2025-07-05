#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief CUDA kernel to apply ReLU activation using float2 + grid-stride loop.
 * 
 * @param input Pointer to input tensor
 * @param output Pointer to output tensor
 * @param N     Total number of elements
 */
__global__
void relu_kernel_vec2(const float* __restrict__ input, float* __restrict__ output, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int vecN = N >> 1;  // Equivalent to N / 2, but faster

    // Process 2 elements at a time
    for (int idx = tid; idx < vecN; idx += stride)
    {
        float2 val = reinterpret_cast<const float2*>(input)[idx];
        val.x = fmaxf(0.0f, val.x);
        val.y = fmaxf(0.0f, val.y);
        reinterpret_cast<float2*>(output)[idx] = val;
    }

    // Handle last element if N is odd
    if (tid == 0 && (N & 1))  // N & 1 is equivalent to N % 2 != 0
    {
        output[N - 1] = fmaxf(0.0f, input[N - 1]);
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
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");

    int N = input.numel();
    
    // Optimize thread/block configuration
    int threads = 512;
    int blocks = min((N + 1023) >> 10, 2048);  // Equivalent to (N + threads - 1) / threads, but faster
    
    // Ensure minimum blocks for GPU utilization
    blocks = max(blocks, 32);

    if (in_place)
    {
        relu_kernel_vec2<<<blocks, threads>>>(
            input.data_ptr<float>(),
            input.data_ptr<float>(),
            N
        );
        return input;
    }
    else
    {
        torch::Tensor output = torch::empty_like(input);
        relu_kernel_vec2<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            N
        );
        return output;
    }
}