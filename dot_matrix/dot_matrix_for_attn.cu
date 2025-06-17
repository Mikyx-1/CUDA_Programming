#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__
void dot_matrix_kernel(const float* a, const float* b, float* c, int batch_size, int seq_len, int dim)
{   

    int batch_id = blockIdx.z;

    int seq_id1 = blockIdx.x * blockDim.x + threadIdx.x;
    int seq_id2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_id < batch_size && seq_id1 < seq_len && seq_id2 < seq_len)
    {
        float sum = 0.0f;
        for (int dim_id = 0; dim_id < dim; dim_id++)
        {   int a_idx = batch_id * seq_len * dim + seq_id1 * dim + dim_id;
            int b_idx = batch_id * seq_len * dim + dim_id * seq_len + seq_id2;
            sum += a[a_idx] * b[b_idx];
        }

        c[batch_id * seq_len * seq_len + seq_id1 * seq_len + seq_id2] = sum;
    }
}


torch::Tensor dot_matrix(torch::Tensor a, torch::Tensor b)
{

    int batch_size = a.size(0);

    int seq_len = a.size(1);
    int dim = a.size(2);


    auto c = torch::zeros({batch_size, seq_len, seq_len}, a.options());

    dim3 threads(16, 16);
    dim3 blocks((seq_len + threads.x - 1) / threads.x, 
                (seq_len + threads.y - 1) / threads.y,
                batch_size);


    dot_matrix_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        batch_size,
        seq_len,
        dim
    );

    return c;
}
