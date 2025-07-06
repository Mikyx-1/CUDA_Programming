#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// A hypothetical __device__ helper function
__device__ __forceinline__ float apply_relu_activation(float value) {
    return fmaxf(0.0f, value);
}

/**
 * @brief CUDA kernel to apply ReLU activation element-wise with striding.
 *
 * @param input Pointer to input tensor
 * @param output Pointer to output tensor
 * @param N     Total number of elements
 */
__global__
void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride)
    {
        // Calling the __device__ helper function here
        output[i] = apply_relu_activation(input[i]);
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
torch::Tensor relu(torch::Tensor input, bool in_place = false)
{
    int N = input.numel();
    int threads = 512;  // use more threads per block for better utilization
    int blocks = (N + threads - 1) / threads;
    // limit max blocks to avoid oversubscribing
    blocks = std::min(blocks, 65535);

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
        auto output = torch::empty_like(input);
        relu_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            N
        );
        return output;
    }
}