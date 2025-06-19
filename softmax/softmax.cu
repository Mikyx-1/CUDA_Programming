#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>



/**
 * @brief Computes the softmax over the last dimension for each batch
 * 
 * @param input Pointer to input tensor data of shape (batch_size, dim)
 * @param output    Pointer to output tensor data (same shape as input)
 * @param batch_size Number of rows (batches)
 * @param dim   Size of the softmax dimension
 * 
 * Each thread computes the softmax for one batch element.
*/
__global__
void softmax_kernel(const float* input, float* output, int batch_size, int dim)
{
    int batch_idx = threadIdx.x;

    if (batch_idx < batch_size)
    {
        float sum_exp = 0.0f;
        for (int i = 0; i < dim; i++)
        {
            float val = __expf(input[batch_idx * dim + i]);
            output[batch_idx * dim + i] = val;
            sum_exp += val;
        }

        for (int i = 0; i < dim; i++)
        {
            output[batch_idx * dim + i] /= sum_exp;
        }
    }
}




/**
 * @brief CUDA wrapper function to compute softmax over the last dimension.
 * 
 * @param input A 2D tensor (batch_size, dim)
 * @return      A tensor of the same shape with softmax applied.
*/
torch::Tensor softmax(torch::Tensor input)
{

    int batch_size = input.size(0);
    int dim = input.size(1);
    torch::Tensor output = torch::empty_like(input);

    dim3 threads(batch_size);
    dim3 blocks(1);

    softmax_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );

    return output;
}