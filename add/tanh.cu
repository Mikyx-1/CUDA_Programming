#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

/**
 * @brief CUDA kernel to apply tanh activation element-wise.
 * 
 * @param input   Pointer to input tensor (device)
 * @param output  Pointer to output tensor (device)
 * @param N       Total number of elements in the tensor
 * 
 * Computes: output[i] = tanh(input[i]) using CUDA’s fast tanhf().
 */
__global__
void tanh_kernel(const float* input, float* output, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N)
    {
        output[tid] = tanhf(input[tid]);  // fast CUDA version of tanh
    }
}

/**
 * @brief Applies hyperbolic tangent activation to a tensor, optionally in-place.
 * 
 * @param input     A tensor of any shape
 * @param in_place  If true, modifies the input tensor directly (default: false)
 * 
 * @return A tensor of the same shape with tanh activation applied.
 * 
 * Formula: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 * 
 * Note: This uses the CUDA-optimized tanhf() function.
 */
torch::Tensor tanh(torch::Tensor input, bool in_place = false)
{
    int N = input.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    if (in_place)
    {
        tanh_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            input.data_ptr<float>(),
            N
        );
        return input;
    }
    else
    {
        torch::Tensor output = torch::empty_like(input);
        tanh_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            N
        );
        return output;
    }
}
