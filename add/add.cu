#include <torch/types.h>
#include <cuda_runtime.h>
#include <cuda.h>


__global__ void add_kernel(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}


torch::Tensor add(torch::Tensor a, torch::Tensor b)
{
    int N = a.numel();

    dim3 threads(256);
    dim3 blocks((N + threads.x - 1) / threads.x);

    torch::Tensor c = torch::empty_like(a);
    add_kernel<<<blocks, threads>>>(a.data_ptr<float>(),
                                    b.data_ptr<float>(),
                                    c.data_ptr<float>(),
                                    N);
    return c;
}