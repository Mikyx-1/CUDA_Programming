#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__
void qkt_kernel(const float* q, const float* k, float* qkt, float scale, int batch_size, int seq_len, int dim)
{

    int batch_id = blockIdx.z;

    int seq_id1 = blockIdx.x * blockDim.x + threadIdx.x;
    int seq_id2 = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_id < batch_size && seq_id1 < seq_len && seq_id2 < seq_len)
    {
        float sum = 0.0f;
        for (int dim_id = 0; dim_id < dim; dim_id++)
        {
            int a_idx = batch_id * seq_len * dim + seq_id1 * dim + dim_id;
            int b_idx = batch_id * seq_len * dim + seq_id2 * dim + dim_id;
            sum += q[a_idx] * k[b_idx];
        }
        qkt[batch_id * seq_len * seq_len + seq_id1 * seq_len + seq_id2] = sum * scale;
    }
}





torch::Tensor self_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{

    int batch_size = Q.size(0);
    int seq_len = Q.size(1);
    int dim_k = Q.size(2);
    int dim_v = V.size(2);

    float scale = 1.0f / (sqrtf(static_cast<float>(dim_k)));

    torch::Tensor qkt = torch::empty({batch_size, seq_len, seq_len}, Q.options());

    // Step 1: Calculate QK^T / scale
    dim3 qkt_threads(16, 16);
    dim3 qkt_blocks((seq_len + qkt_threads.x - 1) / qkt_threads.x,
                    (seq_len + qkt_threads.y - 1) / qkt_threads.y,
                    batch_size);


    qkt_kernel<<<qkt_blocks, qkt_threads>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        qkt.data_ptr<float>(),
        scale,
        batch_size,
        seq_len,
        dim_k
    );

    // Step 2: softmax(QK^T / scale)
    return qkt;
}