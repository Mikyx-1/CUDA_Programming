#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__
void softmax_kernel(const float* input, float* output, int batch_size, int seq_len, int dim)
{
    int batch_idx = blockIdx.x;
    int seq_id = blockIdx.y * blockDim.y + threadIdx.x;

    // Each thread handles one seq_len
    float denom = 0.0f;
    for (int i = 0; i < dim; i++)
    {
        denom += __expf(input[batch_idx * seq_len * dim + seq_id * dim + i]);
    }


    for (int i = 0; i < dim; i++)
    {
        int idx = batch_idx * seq_len * dim + seq_id * dim + i;
        output[idx] = __expf(input[idx]) / denom;
    }
}


torch::Tensor softmax(torch::Tensor input)
{
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int dim = input.size(2);

    torch::Tensor output = torch::empty_like(input);


    dim3 threads(16);
    dim3 blocks(batch_size,
                (seq_len + threads.x - 1) / threads.x);


    softmax_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        dim
    );

    return output;
}