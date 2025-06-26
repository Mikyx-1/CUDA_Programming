#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

/**
 * @brief CUDA kernel for FlashAttention using shared memory optimization
 * 
 * @param q Query tensor [batch_size, seq_len, head_dim]
 * @param k Key tensor [batch_size, seq_len, head_dim]
 * @param v Value tensor [batch_size, seq_len, head_dim]
 * @param output Output tensor [batch_size, seq_len, head_dim]
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param head_dim Head dimension
 * @param block_size Block size for tiling
 * @param scale Scaling factor (1/sqrt(head_dim))
 */
__global__
void flash_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int head_dim,
    int block_size,
    float scale
)
{
    // Block and thread indices
    int batch_idx = blockIdx.z;
    int q_block_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    // Calculate query block boundaries
    int q_start = q_block_idx * block_size;
    int q_end = min(q_start + block_size, seq_len);
    int q_len = q_end - q_start;
    
    if (q_len <= 0) return;
    
    // Shared memory allocation
    extern __shared__ float smem[];
    float* q_smem = smem;
    float* k_smem = q_smem + block_size * head_dim;
    float* v_smem = k_smem + block_size * head_dim;
    float* scores = v_smem + block_size * head_dim;
    
    // Per-thread registers for accumulation
    float acc[32]; // Assuming head_dim <= 32, adjust as needed
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    
    // Initialize accumulator
    for (int d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }
    
    // Load Q block into shared memory
    for (int i = tid; i < q_len * head_dim; i += blockDim.x) {
        int row = i / head_dim;
        int col = i % head_dim;
        if (q_start + row < seq_len && col < head_dim) {
            int global_idx = batch_idx * seq_len * head_dim + (q_start + row) * head_dim + col;
            q_smem[i] = q[global_idx];
        } else {
            q_smem[i] = 0.0f;
        }
    }
    __syncthreads();
    
    // Process each K,V block
    for (int k_start = 0; k_start < seq_len; k_start += block_size) {
        int k_end = min(k_start + block_size, seq_len);
        int k_len = k_end - k_start;
        
        if (k_len <= 0) continue;
        
        // Load K and V blocks
        for (int i = tid; i < k_len * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            int col = i % head_dim;
            if (k_start + row < seq_len && col < head_dim) {
                int global_idx = batch_idx * seq_len * head_dim + (k_start + row) * head_dim + col;
                k_smem[i] = k[global_idx];
                v_smem[i] = v[global_idx];
            } else {
                k_smem[i] = 0.0f;
                v_smem[i] = 0.0f;
            }
        }
        __syncthreads();
        
        // Each thread processes one query row
        if (tid < q_len) {
            // Compute attention scores for this query row
            float local_max = -INFINITY;
            
            // Compute Q[tid] @ K^T and find local max
            for (int k_idx = 0; k_idx < k_len; k_idx++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q_smem[tid * head_dim + d] * k_smem[k_idx * head_dim + d];
                }
                score *= scale;
                scores[tid * block_size + k_idx] = score;
                local_max = fmaxf(local_max, score);
            }
            
            // Compute exp scores and local sum
            float local_sum = 0.0f;
            for (int k_idx = 0; k_idx < k_len; k_idx++) {
                float exp_score = expf(scores[tid * block_size + k_idx] - local_max);
                scores[tid * block_size + k_idx] = exp_score;
                local_sum += exp_score;
            }
            
            // Update global max and renormalize
            float prev_max = row_max;
            float new_max = fmaxf(prev_max, local_max);
            float exp_prev = expf(prev_max - new_max);
            float exp_local = expf(local_max - new_max);
            
            // Compute weighted values for this block
            float local_out[32] = {0.0f}; // Initialize to zero
            for (int d = 0; d < head_dim; d++) {
                for (int k_idx = 0; k_idx < k_len; k_idx++) {
                    local_out[d] += scores[tid * block_size + k_idx] * v_smem[k_idx * head_dim + d];
                }
            }
            
            // Update accumulator
            float new_sum = row_sum * exp_prev + local_sum * exp_local;
            for (int d = 0; d < head_dim; d++) {
                acc[d] = acc[d] * exp_prev + local_out[d] * exp_local;
            }
            
            row_max = new_max;
            row_sum = new_sum;
        }
        
        __syncthreads();
    }
    
    // Write final output
    if (tid < q_len) {
        for (int d = 0; d < head_dim; d++) {
            int global_idx = batch_idx * seq_len * head_dim + (q_start + tid) * head_dim + d;
            output[global_idx] = acc[d] / row_sum;
        }
    }
}

/**
 * @brief Host function for FlashAttention
 * 
 * @param q Query tensor [batch_size, seq_len, head_dim]
 * @param k Key tensor [batch_size, seq_len, head_dim]
 * @param v Value tensor [batch_size, seq_len, head_dim]
 * @param block_size Block size for tiling (default: 64)
 * 
 * @return Attention output tensor [batch_size, seq_len, head_dim]
 */
torch::Tensor flash_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int block_size = 64
)
{
    // Check input tensors
    TORCH_CHECK(q.dim() == 3, "Query tensor must be 3D");
    TORCH_CHECK(k.dim() == 3, "Key tensor must be 3D");
    TORCH_CHECK(v.dim() == 3, "Value tensor must be 3D");
    TORCH_CHECK(q.sizes() == k.sizes(), "Q and K must have same shape");
    TORCH_CHECK(q.sizes() == v.sizes(), "Q and V must have same shape");
    TORCH_CHECK(q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "V must be contiguous");
    
    // Get tensor dimensions
    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int head_dim = q.size(2);
    
    // Ensure head_dim is reasonable for register allocation
    TORCH_CHECK(head_dim <= 32, "Head dimension must be <= 32 for this implementation");
    
    // Scale factor
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    // Create output tensor
    torch::Tensor output = torch::zeros_like(q);
    
    // Calculate shared memory size
    int smem_size = (
        3 * block_size * head_dim +     // q_smem, k_smem, v_smem
        block_size * block_size         // scores
    ) * sizeof(float);
    
    // Launch configuration
    int num_q_blocks = (seq_len + block_size - 1) / block_size;
    dim3 grid(1, num_q_blocks, batch_size);
    dim3 block(block_size); // One thread per query in the block
    
    // Check shared memory limit
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (smem_size > prop.sharedMemPerBlock) {
        TORCH_CHECK(false, "Shared memory requirement exceeds device limit. Reduce block_size.");
    }
    
    // Launch kernel
    flash_attention_kernel<<<grid, block, smem_size>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        head_dim,
        block_size,
        scale
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    cudaDeviceSynchronize();
    
    return output;
}