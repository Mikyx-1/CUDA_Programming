#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>  // you had cuda.h twice, cleaned that up

__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d, 
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float* m, float* O)
{
    int thread_id = threadIdx.x;
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;

    int qkv_offset = (batch_id * gridDim.y * N * d) + (head_id * N * d);
    int lm_offset = (batch_id * gridDim.y * N) + (head_id * N);

    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float* Qi = sram; 
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int tile_column_id = 0; tile_column_id < Tc; tile_column_id++)
    {
        int col_start = tile_column_id * Bc;
        int global_col_id = col_start + thread_id;

        // Load Kj, Vj with guard
        if (global_col_id < N) {
            for (int dim_id = 0; dim_id < d; dim_id++) {
                Kj[thread_id * d + dim_id] = K[qkv_offset + global_col_id * d + dim_id];
                Vj[thread_id * d + dim_id] = V[qkv_offset + global_col_id * d + dim_id];
            }
        } else {
            for (int dim_id = 0; dim_id < d; dim_id++) {
                Kj[thread_id * d + dim_id] = 0.0f;
                Vj[thread_id * d + dim_id] = 0.0f;
            }
        }
        __syncthreads();

        for (int tile_row_id = 0; tile_row_id < Tr; tile_row_id++)
        {
            int row_start = tile_row_id * Br;
            int global_row_id = row_start + thread_id;

            // Load Qi with guard
            if (global_row_id < N) {
                for (int dim_id = 0; dim_id < d; dim_id++) {
                    Qi[thread_id * d + dim_id] = Q[qkv_offset + global_row_id * d + dim_id];
                }
            } else {
                for (int dim_id = 0; dim_id < d; dim_id++) {
                    Qi[thread_id * d + dim_id] = 0.0f;
                }
            }

            float row_m_prev = (global_row_id < N) ? m[lm_offset + global_row_id] : -INFINITY;
            float row_l_prev = (global_row_id < N) ? l[lm_offset + global_row_id] : 0.0f;

            // Compute S
            float row_m = -INFINITY;
            for (int column_id = 0; column_id < Bc; column_id++) {
                int global_col_id_inner = tile_column_id * Bc + column_id;
                float sum = 0.0f;
                for (int dim_id = 0; dim_id < d; dim_id++) {
                    sum += Qi[thread_id * d + dim_id] * Kj[column_id * d + dim_id];
                }
                sum *= softmax_scale;
                if (global_col_id_inner < N) {
                    S[thread_id * Bc + column_id] = sum;
                    if (sum > row_m) row_m = sum;
                } else {
                    S[thread_id * Bc + column_id] = -INFINITY;
                }
            }

            // Compute P, row_l
            float row_l = 0.0f;
            for (int column_id = 0; column_id < Bc; column_id++) {
                float val = S[thread_id * Bc + column_id];
                val = __expf(val - row_m);
                S[thread_id * Bc + column_id] = val;
                row_l += val;
            }

            float row_m_new = fmaxf(row_m_prev, row_m);
            float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev
                            + __expf(row_m - row_m_new) * row_l;

            // Accumulate output
            if (global_row_id < N) {
                for (int dim_id = 0; dim_id < d; dim_id++) {
                    float res = 0.0f;
                    for (int column_id = 0; column_id < Bc; column_id++) {
                        res += S[thread_id * Bc + column_id] * Vj[column_id * d + dim_id];
                    }
                    O[qkv_offset + global_row_id * d + dim_id] = (1.0f / row_l_new) *
                        (row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + global_row_id * d + dim_id] +
                         __expf(row_m - row_m_new) * res);
                }
                m[lm_offset + global_row_id] = row_m_new;
                l[lm_offset + global_row_id] = row_l_new;
            }
        }
        __syncthreads();
    }
}

torch::Tensor flash_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{
    const int Bc = 32;
    const int Br = 32;

    const int B = Q.size(0);
    const int nh = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Tc = ceilf((float) N / Bc);
    const int Tr = ceilf((float) N / Br);
    const float softmax_scale = 1.0f / sqrtf((float)d);

    torch::Tensor O = torch::zeros({B, nh, N, d}, Q.options());
    torch::Tensor l = torch::zeros({B, nh, N}, Q.options());
    torch::Tensor m = torch::full({B, nh, N}, -INFINITY, Q.options());

    const int sram_size = (3 * Bc * d + Bc * Br) * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);
    dim3 block_dim(Bc);

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale, 
        l.data_ptr<float>(),
        m.data_ptr<float>(),
        O.data_ptr<float>()
    );

    return O;
}