#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__
void self_attention_kernel(const float* Q, const float* K, const float* V, float* output,
                           int batch_size, int seq_len, int dim, float scale)
{
    extern __shared__ float shared[];
    float* qk_scores = shared;
    float* softmax_weights = shared + seq_len;

    int batch_id = blockIdx.x;
    int query_id = blockIdx.y;
    int dim_id = threadIdx.x;

    if (batch_id >= batch_size || query_id >= seq_len || dim_id >= dim)
        return;

    const float* Q_batch = Q + batch_id * seq_len * dim;
    const float* K_batch = K + batch_id * seq_len * dim;
    const float* V_batch = V + batch_id * seq_len * dim;


    const float* current_query = Q_batch + query_id * dim;

    // Step 1: Compute attention scores QK^T

    for (int key_id = 0; key_id < seq_len; key_id++)
    {
        const float* current_key = K_batch + key_id * dim;


        float thread_contrib = 0.0f;
        if (dim_id < dim)
        {
            thread_contrib = current_query[dim_id] * current_key[dim_id];
        }


        __shared__ float reduction_buffer[1024];
        reduction_buffer[dim_id] = thread_contrib;

        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {   
            if (threadIdx.x < stride)
                reduction_buffer[threadIdx.x] += reduction_buffer[threadIdx.x + stride];
            __syncthreads();    // This fixes the issue
        }


        qk_scores[key_id] = reduction_buffer[0] * scale;

    }

    __syncthreads();

    // Step 2: Compute softmax of attention scores
    if (threadIdx.x == 0)
    {
        // Find maximum value for numerical stability
        float max_score = -INFINITY;
        for (int i = 0; i < seq_len; i++)
        {
            max_score = fmaxf(max_score, qk_scores[i]);
        }

        
        // Compute exp(score - max) and sum for normalisation
        float sum_exp = 0.0f;
        for (int i = 0; i < seq_len; i++)
        {
            softmax_weights[i] = __expf(qk_scores[i] - max_score);
            sum_exp += softmax_weights[i];
        }


        // Normalise
        for (int i = 0; i < seq_len; i++)
        {
            softmax_weights[i] /= sum_exp;
        }
    }

    __syncthreads();

    // Step 3: Compute weighted sum of values

    if (dim_id < dim)
    {
        float output_val = 0.0f;

        for (int i = 0; i < seq_len; i++)
        {
            output_val += softmax_weights[i] * V_batch[i*dim + dim_id];
        }
        output[batch_id * seq_len * dim + query_id * dim + dim_id] = output_val;
    }
}





torch::Tensor self_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{

    int batch_size = Q.size(0);
    int seq_len = Q.size(1);
    int dim = Q.size(2);

    float scale = 1.0f / sqrtf(static_cast<float>(dim));

    torch::Tensor output = torch::empty_like(Q);


    dim3 threads(min(dim, 1024));
    dim3 blocks(batch_size, seq_len);

    int shared_mem_size = (2 * seq_len + 1024) * sizeof(float);

    // Launch kernel
    self_attention_kernel<<<blocks, threads, shared_mem_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        dim,
        scale
    );

    return output;
}