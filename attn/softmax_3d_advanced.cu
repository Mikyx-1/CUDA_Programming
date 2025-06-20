#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32


__global__
void softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int batch_size, int seq_len, int dim)
{
    // Each block handles one (batch_idx, seq_idx) pair
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Compute the flat offset for input/output pointer
    int base_idx = (batch_idx * seq_len + seq_idx) * dim;

    // Shared memory for partial max and sum
    __shared__ float tile_max;
    __shared__ float tile_sum;

    // Step 1: compute max for numerical stability
    float local_max = -INFINITY;
    for (int i = tid; i < dim; i += stride)
    {
        float val = input[base_idx + i];
        local_max = fmaxf(local_max, val);
    }

    // Reduce to get global max in the tile
    float max_val = local_max;
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));

    if (tid == 0) tile_max = max_val;
    __syncthreads();

    // Step 2: compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += stride)
    {
        float val = expf(input[base_idx + i] - tile_max);
        local_sum += val;
        output[base_idx + i] = val; // Temporarily store unnormalized
    }

    // Reduce to get global sum in the tile
    float sum_val = local_sum;
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);

    if (tid == 0) tile_sum = sum_val;
    __syncthreads();

    // Step 3: normalize
    for (int i = tid; i < dim; i += stride)
    {
        output[base_idx + i] /= tile_sum;
    }
}


torch::Tensor softmax(torch::Tensor input)
{
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int dim = input.size(2);


    torch::Tensor output = torch::empty_like(input);

    dim3 threads(TILE_SIZE);
    dim3 blocks(batch_size, seq_len);

    softmax_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        dim
    );

    return output;
}