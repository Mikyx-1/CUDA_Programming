#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
 * FlashAttention CUDA Kernel Implementation
 * 
 * This implements the FlashAttention algorithm which computes attention efficiently
 * by tiling the computation and keeping intermediate results in fast SRAM (shared memory)
 * instead of slow HBM (global memory).
 * 
 * Key innovation: Instead of materializing the full N×N attention matrix,
 * we process it in tiles and use online softmax to maintain numerical stability.
 * 
 * Algorithm overview:
 * 1. Divide Q into Tr row blocks, K,V into Tc column blocks
 * 2. For each block of Q, iterate through all blocks of K,V
 * 3. Compute attention scores for the current Q,K block pair
 * 4. Use online softmax algorithm to incrementally update output
 */

__global__
void forward_kernel(const float* Q,           // Query matrix [B, nh, N, d]
                    const float* K,           // Key matrix [B, nh, N, d] 
                    const float* V,           // Value matrix [B, nh, N, d]
                    const int N,              // Sequence length
                    const int d,              // Head dimension
                    const int Tc,             // Number of column tiles (for K,V)
                    const int Tr,             // Number of row tiles (for Q)
                    const int Bc,             // Column block size
                    const int Br,             // Row block size
                    const float softmax_scale, // 1/sqrt(d) for attention scaling
                    float* l,                 // Row sums for softmax normalization [B, nh, N]
                    float* m,                 // Row maxes for numerical stability [B, nh, N]
                    float* O) {               // Output matrix [B, nh, N, d]
    
    // Thread and block indices
    int tx = threadIdx.x;           // Thread index within block (0 to Bc-1)
    int bx = blockIdx.x;           // Batch index
    int by = blockIdx.y;           // Head index

    // Calculate memory offsets for this specific batch and head
    // Each batch×head combination processes an N×d matrix
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh (num_heads)
    int lm_offset = (bx * gridDim.y * N) + (by * N);          // offset for l and m vectors

    // Allocate shared memory (SRAM) for tile storage
    // This is much faster than global memory and is the key to FlashAttention's efficiency
    extern __shared__ float sram[];
    int tile_size = Bc * d;        // Size of each tile in elements
    
    // Partition shared memory into different regions:
    float* Qi = sram;                    // Current Q tile [Bc, d]
    float* Kj = &sram[tile_size];        // Current K tile [Bc, d] 
    float* Vj = &sram[tile_size * 2];    // Current V tile [Bc, d]
    float* S = &sram[tile_size * 3];     // Attention scores [Bc, Bc]

    // Outer loop: iterate through all K,V tiles (columns of attention matrix)
    for (int j = 0; j < Tc; j++) {

        // Load the j-th tile of K and V from global memory to shared memory
        // Each thread loads d elements (one row of the tile)
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // Ensure all threads finish loading before proceeding

        // Inner loop: iterate through all Q tiles (rows of attention matrix)
        for (int i = 0; i < Tr; i++) {

            // Load the i-th tile of Q from global memory to shared memory
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            
            // Load previous running statistics for this row
            // These are needed for the online softmax algorithm
            float row_m_prev = m[lm_offset + (Br * i) + tx];  // Previous row maximum
            float row_l_prev = l[lm_offset + (Br * i) + tx];  // Previous row sum

            // Compute attention scores S = Q_i × K_j^T for current tile pair
            // Each thread computes one row of the attention block
            float row_m = -INFINITY;  // Track maximum for numerical stability
            
            for (int y = 0; y < Bc; y++) {  // For each column in current K tile
                float sum = 0;
                // Dot product between current Q row and current K row
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;  // Scale by 1/sqrt(d)
                S[(Bc * tx) + y] = sum;
                
                // Track row maximum for numerical stability in softmax
                if (sum > row_m)
                    row_m = sum;
            }

            // Compute softmax probabilities P = exp(S - row_m)
            // Subtracting row_m prevents overflow in exponential
            float row_l = 0;  // Sum of probabilities in this row
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Online softmax update: combine current tile stats with previous stats
            // This is the key insight that allows processing tiles independently
            float row_m_new = max(row_m_prev, row_m);  // New maximum
            // New sum: rescale both previous and current sums by their relative exp differences
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + 
                              (__expf(row_m - row_m_new) * row_l);

            // Update output O using online softmax formula
            // O_new = (1/l_new) * (l_prev * exp(m_prev - m_new) * O_prev + exp(m - m_new) * P*V)
            for (int x = 0; x < d; x++) {
                // Compute P*V for current tile (attention-weighted values)
                float pv = 0;
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                
                // Combine previous output with current tile contribution
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) * 
                    ((row_l_prev * __expf(row_m_prev - row_m_new) * 
                      O[qkv_offset + (tile_size * i) + (tx * d) + x]) + 
                     (__expf(row_m - row_m_new) * pv));
            }
            
            // Update running statistics for next iteration
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // Ensure all threads finish before loading next K,V tile
    }
}

/*
 * Host function to launch the FlashAttention kernel
 * 
 * This function sets up the grid configuration, allocates auxiliary memory,
 * and launches the CUDA kernel to compute attention.
 */
torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // Block sizes - these determine the tile dimensions
    // TODO: These should be determined dynamically based on available shared memory
    const int Bc = 32; // Column block size (for K,V tiles)
    const int Br = 32; // Row block size (for Q tiles)

    // Extract tensor dimensions
    const int B = Q.size(0);   // Batch size
    const int nh = Q.size(1);  // Number of attention heads
    const int N = Q.size(2);   // Sequence length
    const int d = Q.size(3);   // Head dimension

    // Calculate number of tiles needed
    const int Tc = ceil((float) N / Bc);  // Number of column tiles
    const int Tr = ceil((float) N / Br);  // Number of row tiles
    const float softmax_scale = 1.0 / sqrt(d);  // Attention scaling factor

    // Initialize output and auxiliary tensors
    auto O = torch::zeros_like(Q);              // Output tensor
    auto l = torch::zeros({B, nh, N});          // Row sums (for softmax normalization)
    auto m = torch::full({B, nh, N}, -INFINITY); // Row maxes (for numerical stability)
    
    // Move auxiliary tensors to GPU
    torch::Device device(torch::kCUDA);
    l = l.to(device); 
    m = m.to(device);

    // Calculate shared memory requirements
    // Need space for: Qi[Bc×d] + Kj[Bc×d] + Vj[Bc×d] + S[Bc×Bc]
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    
    // Check if we have enough shared memory
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    // Configure grid and block dimensions
    dim3 grid_dim(B, nh);    // One block per (batch, head) pair
    dim3 block_dim(Bc);      // Bc threads per block (one per row in tile)

    // Launch the kernel
    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    
    return O;
}