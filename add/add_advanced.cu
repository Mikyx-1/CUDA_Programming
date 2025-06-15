#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>



#define TILE_SIZE 256

__global__ void tiled_add_kernel(const float* A, const float* B, float* C, int N)
{
    __shared__ float TILE_A[TILE_SIZE];
    __shared__ float TILE_B[TILE_SIZE];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if (idx < N)
    {
        TILE_A[tid] = A[idx];
        TILE_B[tid] = B[idx];
    }

    __syncthreads();

    if (idx < N)
    {
        C[idx] = TILE_A[tid] + TILE_B[tid];
    }
}


torch::Tensor add(torch::Tensor a, torch::Tensor b)
{
    int N = a.numel();
    torch::Tensor c = torch::empty_like(a);

    dim3 threads(TILE_SIZE);
    dim3 blocks((N + TILE_SIZE -1) / TILE_SIZE);

    tiled_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        N
    );

    return c;
}