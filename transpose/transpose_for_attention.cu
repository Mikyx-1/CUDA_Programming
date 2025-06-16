#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void transpose_kernel(const float* input, float* output, int batch_size, int seq_len, int dim)
{

    int batch_id = blockIdx.x;

    // Get thread ID within the block
    int seq_idx = threadIdx.x + blockIdx.y * blockDim.x;
    int dim_idx = threadIdx.y + blockIdx.z * blockDim.y;

    if (batch_id < batch_size && seq_idx < seq_len && dim_idx < dim)
    {
        int orig_idx = batch_id * seq_len * dim + seq_idx * dim + dim_idx;
        int transposed_idx = batch_id * dim * seq_len + dim_idx * seq_len + seq_idx;

        output[transposed_idx] = input[orig_idx];
    }
}


torch::Tensor transpose(torch::Tensor input)
{
    int batch_size = input.size(0); 
    int seq_len = input.size(1);
    int dim = input.size(2);

    torch::Tensor output = torch::empty({batch_size, dim, seq_len}, input.options());


    // Set threads per block
    dim3 threads(16, 16);

    // Calculate number of blocks needed to cover entire output
    dim3 blocks(batch_size, (seq_len + threads.x - 1) / threads.x,
                (dim + threads.y - 1) / threads.y);

    transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        dim
    );

    return output;
}