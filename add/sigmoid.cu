#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>



/**
 * @brief CUDA kernel to apply sigmoid activation element-wise.
 * 
 * @param input   Pointer to input tensor
 * @param output  Pointer to output tensor
 * @param N       Total number of elements
 * 
 * Computes: output[i] = 1 / (1 + exp(-input[i]))
 */
__global__
void sigmoid_kernel(const float* input, float* output, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N)
    {
        output[tid] = 1.0f / (1.0f + expf(-input[tid]));
    }
}


/**
 * @brief Applies sigmoid activation to a tensor, optionally in-place.
 * 
 * @param input     A tensor of any shape
 * @param in_place  If true, modifies the input tensor directly (default: false)
 * 
 * @return A tensor of the same shape with sigmoid applied
 * 
 * Example: y = sigmoid(x) = 1 / (1 + exp(-x))
 */
torch::Tensor sigmoid(torch::Tensor input, bool in_place = False)
{
    int N = input.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    if (in_place == true)
    {
        sigmoid_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            input.data_ptr<float>(),
            N
        );
    }
    else
    {
        torch::Tensor output = torch::empty_like(input);

        sigmoid_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            N
        );
    }

    return output;
}