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


#define SOFTMAX_TILE_SIZE 32


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
    __shared__ float shared_data[SOFTMAX_TILE_SIZE];

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


__global__
void dot_matrix(const float* a, const float* b, float* c, int batch_size, int a_height, int a_width, int b_height, int b_width)
{
    // Ensure a_width == b_height
    // Dot 2 matrices of size (batch_size, a_height, a_width) (batch_size, b_height, b_width)
    // Expected output (batch_size, a_height, b_width)
    
    int batch_id = blockIdx.z;

    int ay = blockIdx.x * blockDim.x + threadIdx.x;
    int bx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_id < batch_size && ay < a_height && bx < b_width)
    {   
        float sum = 0.0f;
        for (int dim_id = 0; dim_id < a_width; dim_id++)
        {
            int a_idx = batch_id * a_height * a_width + ay * a_width + dim_id;
            int b_idx = batch_id * b_height * b_width + dim_id * b_width + bx;
            sum += a[a_idx] * b[b_idx];
        }
        c[batch_id * a_height * b_width + ay * b_width + bx] = sum;
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
    torch::Tensor attn_scores = torch::empty_like(qkt);
    torch::Tensor output = torch::empty_like(V);

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
    dim3 softmax_threads(SOFTMAX_TILE_SIZE);
    dim3 softmax_blocks(batch_size, seq_len);

    softmax_kernel<<<softmax_blocks, softmax_threads>>>(
        qkt.data_ptr<float>(),
        attn_scores.data_ptr<float>(),
        batch_size,
        seq_len,
        seq_len
    );

    // Step 3: softmax(QK^T / scale)V  (Not implemented yet)

    dim3 dot_matrix_threads(16, 16);
    dim3 dot_matrix_blocks((seq_len + dot_matrix_threads.x - 1) / dot_matrix_threads.x,
                           (dim_v + dot_matrix_threads.y - 1) / dot_matrix_threads.y,
                           batch_size);

    dot_matrix<<<dot_matrix_blocks, dot_matrix_threads>>>(
        attn_scores.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        seq_len,
        seq_len,
        dim_v
    );
    return output;
}
