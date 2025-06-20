#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define TILE_SIZE 32

__global__
void softmax_kernel(const float* input, float* output, int batch_size, int seq_len, int dim)
{
    // Each block handles one (batch_idx, seq_idx) pair
    int batch_idx = blockIdx.x; 
    int seq_idx = blockIdx.y;

    int tid = threadIdx.x;

    // Compute the flat offset for input/output pointer
    int base_idx = batch_idx * seq_len * dim + seq_idx * dim;


    // Shared memory initialisation
    __shared__ float shared_data[TILE_SIZE];

    // Step 1: Compute max for numerical stability
    float local_max = -INFINITY;
    for (int i = tid; i < dim; i += blockDim.x)
    {
        local_max = fmaxf(local_max, input[base_idx + i]);
    }
    shared_data[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
    }

    __syncthreads();
    float row_max = shared_data[0];

    // Step 2: Compute exp(x - row_max) and sum
    float local_denom = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x)
    {
        float val = __expf(input[base_idx + i] - row_max);
        output[base_idx + i] = val;
        local_denom += val;
    }
    shared_data[tid] = local_denom;
    __syncthreads();


    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared_data[tid] += shared_data[tid + stride];
        }
    }

    __syncthreads();
    float global_denom = shared_data[0];

    // Step 3: Normalise
    for (int i = tid; i < dim; i+= blockDim.x)
    {
        output[base_idx + i] /= global_denom;
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