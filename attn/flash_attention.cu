#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for FlashAttention forward pass
__global__
void forward_kernel(const float* Q, const float* K, const float* V,
                    const int N, const int d,
                    const int num_col_tiles, const int num_row_tiles,
                    const int col_block_size, const int row_block_size,
                    const float softmax_scale,
                    float* l, float* m, float* O)
{
    const int thread_id = threadIdx.x;
    const int batch_id  = blockIdx.x;
    const int head_id   = blockIdx.y;

    const int qkv_offset = (batch_id * gridDim.y * N * d) + (head_id * N * d);
    const int lm_offset  = (batch_id * gridDim.y * N) + (head_id * N);

    extern __shared__ float shared_mem[];
    const int tile_mem = col_block_size * d;
    float* Qi = shared_mem;
    float* Kj = &shared_mem[tile_mem];
    float* Vj = &shared_mem[2 * tile_mem];
    float* S  = &shared_mem[3 * tile_mem]; // Br x Bc matrix

    for (int j = 0; j < num_col_tiles; ++j) {

        // Load current K and V tiles to shared memory
        for (int k = 0; k < d; ++k) {
            Kj[thread_id * d + k] = K[qkv_offset + j * tile_mem + thread_id * d + k];
            Vj[thread_id * d + k] = V[qkv_offset + j * tile_mem + thread_id * d + k];
        }
        __syncthreads();

        for (int i = 0; i < num_row_tiles; ++i) {

            // Load Q tile to shared memory
            for (int k = 0; k < d; ++k) {
                Qi[thread_id * d + k] = Q[qkv_offset + i * tile_mem + thread_id * d + k];
            }

            float prev_m = m[lm_offset + i * row_block_size + thread_id];
            float prev_l = l[lm_offset + i * row_block_size + thread_id];

            // Compute raw attention scores S = Q.K^T (scaled)
            float max_val = -INFINITY;
            for (int y = 0; y < col_block_size; ++y) {
                float dot = 0.0f;
                for (int k = 0; k < d; ++k) {
                    dot += Qi[thread_id * d + k] * Kj[y * d + k];
                }
                dot *= softmax_scale;
                S[thread_id * col_block_size + y] = dot;
                max_val = fmaxf(max_val, dot);
            }

            float row_l = 0.0f;
            for (int y = 0; y < col_block_size; ++y) {
                S[thread_id * col_block_size + y] = __expf(S[thread_id * col_block_size + y] - max_val);
                row_l += S[thread_id * col_block_size + y];
            }

            float new_m = fmaxf(prev_m, max_val);
            float new_l = __expf(prev_m - new_m) * prev_l + __expf(max_val - new_m) * row_l;

            for (int k = 0; k < d; ++k) {
                float val_sum = 0.0f;
                for (int y = 0; y < col_block_size; ++y) {
                    val_sum += S[thread_id * col_block_size + y] * Vj[y * d + k];
                }

                float prev_O = O[qkv_offset + i * tile_mem + thread_id * d + k];
                float updated_O = (1.0f / new_l) *
                    (__expf(prev_m - new_m) * prev_l * prev_O + __expf(max_val - new_m) * val_sum);

                O[qkv_offset + i * tile_mem + thread_id * d + k] = updated_O;
            }

            m[lm_offset + i * row_block_size + thread_id] = new_m;
            l[lm_offset + i * row_block_size + thread_id] = new_l;
        }
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int col_block_size = 32;
    const int row_block_size = 32;

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Tc = (N + col_block_size - 1) / col_block_size;
    const int Tr = (N + row_block_size - 1) / row_block_size;
    const float scale = 1.0f / sqrtf((float)d);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, H, N}, Q.options());
    auto m = torch::full({B, H, N}, -INFINITY, Q.options());

    const int shared_mem_size = (3 * col_block_size * d + col_block_size * row_block_size) * sizeof(float);

    dim3 grid_dim(B, H);
    dim3 block_dim(col_block_size);

    forward_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, col_block_size, row_block_size, scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );

    return O;
}
