#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


/**
 * @brief Multiply tensor a by tensor b
 * 
 * @param a Pointer to tensor a data. 
 * @param b Pointer to tensor b data.
 * @param c Pointer to tensor c data.
 * @param N number of elements 
 * 
 * Each thread computes the multiplication of 2 elements in one position.
*/
__global__
void mul_kernel(const float* a, const float* b, float* c, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N)
    {
        c[tid] = a[tid] * b[tid];
    }
}


/**
 * @brief CUDA wrapper function to compute multipliation of 2 tensors.
 * 
 * @param a torch::Tensor a
 * @param b torch::Tensor b
 * 
 * @return a tensor of the same shape (a * b)
*/
torch::Tensor mul(torch::Tensor a, torch::Tensor b)
{
    
    torch::Tensor c = torch::empty_like(a);

    int N = a.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    mul_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        N
    );
}