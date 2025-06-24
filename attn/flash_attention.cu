#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Main kernel: computes attention output `O` using Q, K, V (query, key, value) with tiling and stable softmax.
__global__
void forward_kernel(const float* Q, const float* K, const float* V, 
                    const int N, const int d,               // N: sequence length, d: embedding dim
                    const int Tc, const int Tr,             // Tc: #column tiles, Tr: #row tiles
                    const int Bc, const int Br,             // Bc: column block size, Br: row block size
                    const float softmax_scale,              // scaling factor for attention
                    float* l, float* m, float* O)           // softmax accumulators (l,m) and output (O)
{
    int tx = threadIdx.x;      // Thread index within the block (0 .. Bc-1)
    int bx = blockIdx.x;       // Batch index
    int by = blockIdx.y;       // Head index

    // Indexing offset for Q, K, V, and O tensors: [B, H, N, d] flattened
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    int lm_offset  = (bx * gridDim.y * N) + (by * N); // for l and m: [B, H, N]

    // Shared memory layout:
    // - Qi: Br x d
    // - Kj: Bc x d
    // - Vj: Bc x d
    // - S : Br x Bc
    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float* Qi = sram;                         // Qi tile: Br rows of Q
    float* Kj = &sram[tile_size];             // Kj tile: Bc rows of K
    float* Vj = &sram[2 * tile_size];         // Vj tile: Bc rows of V
    float* S  = &sram[3 * tile_size];         // S matrix: (Br x Bc)

    // Loop over all column tiles (K, V)
    for (int j = 0; j < Tc; j++) {

        // === Load K and V tiles to shared memory ===
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (j * tile_size) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (j * tile_size) + (tx * d) + x];
        }
        __syncthreads();  // Make sure all threads finish loading Kj and Vj

        // Loop over all row tiles (Q)
        for (int i = 0; i < Tr; i++) {

            // === Load Qi tile to shared memory ===
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (i * tile_size) + (tx * d) + x];
            }

            // Load previous softmax stats
            float prev_m = m[lm_offset + (Br * i) + tx];
            float prev_l = l[lm_offset + (Br * i) + tx];

            // === Compute raw attention scores: S = Q x K^T ===
            float max_val = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float dot = 0;
                for (int x = 0; x < d; x++) {
                    dot += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                dot *= softmax_scale;
                S[(Bc * tx) + y] = dot;
                if (dot > max_val) max_val = dot;
            }

            // === Compute softmax: P = exp(S - max), accumulate new l, m ===
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - max_val);
                row_l += S[(Bc * tx) + y];
            }

            // Numerically stable softmax accumulation
            float new_m = fmaxf(prev_m, max_val);
            float new_l = __expf(prev_m - new_m) * prev_l + __expf(max_val - new_m) * row_l;

            // === Compute weighted sum of values: O += P @ V ===
            for (int x = 0; x < d; x++) {
                float val_sum = 0;
                for (int y = 0; y < Bc; y++) {
                    val_sum += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }

                // Numerically stable update of O
                float prev_O = O[qkv_offset + (i * tile_size) + (tx * d) + x];
                float updated_O = (1.0f / new_l) *
                    (__expf(prev_m - new_m) * prev_l * prev_O + __expf(max_val - new_m) * val_sum);
                O[qkv_offset + (i * tile_size) + (tx * d) + x] = updated_O;
            }

            // Save new softmax stats to HBM
            m[lm_offset + (Br * i) + tx] = new_m;
            l[lm_offset + (Br * i) + tx] = new_l;
        }

        __syncthreads(); // Avoid reading stale K/V from shared memory
    }
}

// Host wrapper: manages memory and launches the CUDA kernel
torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // Block dimensions: tunable tile sizes
    const int Bc = 32; const int Br = 32;

    // Tensor shapes: Q, K, V ∈ [B, H, N, d]
    const int B  = Q.size(0);
    const int nh = Q.size(1);
    const int N  = Q.size(2);
    const int d  = Q.size(3);

    // Tile count (ceil divisions)
    const int Tc = (N + Bc - 1) / Bc;
    const int Tr = (N + Br - 1) / Br;

    const float softmax_scale = 1.0f / sqrtf((float)d);

    // Allocate output tensor and softmax statistics
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N}).to(Q.device());
    auto m = torch::full({B, nh, N}, -INFINITY).to(Q.device());

    // Compute required shared memory size per block
    const int sram_size = (3 * Bc * d + Bc * Br) * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested: %d\n", max_sram_size, sram_size);

    // Launch kernel
    dim3 grid_dim(B, nh);  // 2D grid: [batch, heads]
    dim3 block_dim(Bc);    // 1D block: Bc threads per row of Q

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );

    return O;
}
