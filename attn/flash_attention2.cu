#include <torch/types.h>
#include <cuda.h>
#include <cuda.h>



__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d, 
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float* m, float* O)
{

    int thread_id = threadIdx.x;    // Seq_id
    int batch_id = blockIdx.x;      // Batch_id
    int head_id = blockIdx.y;       // Head_id

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
        // Load Kj, Vj to SRAM
        for (int dim_id = 0; dim_id < d; dim_id++)
        {
            Kj[(thread_id * d) + dim_id] = K[qkv_offset + (tile_column_id * tile_size) + (thread_id * d) + dim_id];
            Vj[(thread_id * d) + dim_id] = V[qkv_offset + (tile_column_id * tile_size) + (thread_id * d) + dim_id];
            // Print here to see loaded values
        }
        __syncthreads();

        for (int tile_row_id = 0; tile_row_id < Tr; tile_row_id++)
        {
            // Load Qi to SRAM
            for (int dim_id = 0; dim_id < d; dim_id++)
            {
                Qi[(thread_id * d) + dim_id] = Q[qkv_offset + (tile_row_id * tile_size) + (thread_id * d) + dim_id];
            }


            float row_m_prev = m[lm_offset + (tile_row_id * Br) + thread_id];
            float row_l_prev = l[lm_offset + (tile_row_id * Br) + thread_id];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int column_id = 0; column_id < Bc; column_id++)
            {
                float sum = 0.0f;
                for (int dim_id = 0; dim_id < d; dim_id++)
                {
                    sum += Qi[(thread_id * d) + dim_id] * Kj[(column_id * d) + dim_id];
                }
                sum *= softmax_scale;
                S[(Bc * thread_id) + column_id] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0.0f;
            for (int seq_id = 0; seq_id < Bc; seq_id++)
            {
                S[(thread_id * Bc) + seq_id] = __expf(S[(thread_id * Bc) + seq_id] - row_m);
                row_l += S[(thread_id * Bc) + seq_id];
            }

            // Compute new m and l
            float row_m_new = fmaxf(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);


            // Write O, l, m to HBM
            for (int dim_id = 0; dim_id < d; dim_id++)
            {
                float res = 0.0f;
                for (int seq_id = 0; seq_id < Bc; seq_id++)
                {
                    res += S[(thread_id * Bc) + seq_id] * Vj[(seq_id * d) + dim_id];
                }
                O[qkv_offset + (tile_row_id * tile_size) + thread_id * d + dim_id] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * tile_row_id) + (thread_id * d) + dim_id]) \
                    + (__expf(row_m - row_m_new) * res));
            }

            m[lm_offset + (Br * tile_row_id) + thread_id] = row_m_new;
            l[lm_offset + (Br * tile_row_id) + thread_id] = row_l_new;
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


    const int Tc = ceil((float) N / Bc);
    const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0f / sqrt(d);

    // Initialise O, l, m to HBM
    torch::Tensor O = torch::zeros({B, nh, N, d}, Q.options());
    torch::Tensor l = torch::zeros({B, nh, N}, Q.options());
    torch::Tensor m = torch::full({B, nh, N}, -INFINITY, Q.options());

    const int sram_size = (3 * Bc * d  + Bc * Br) * sizeof(float);
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