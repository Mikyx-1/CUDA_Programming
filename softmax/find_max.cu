#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__
void find_max_kernel(const float* input, float* output, int batch_size, int dim)
{


    extern __shared__ float shared_data[];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    const float* input_row = input + batch_idx * dim;

    float max_val = -INFINITY;

    for (int i = tid; i < dim; i+= blockDim.x)
    {
        max_val = fmaxf(max_val, input_row[tid]);
    }

    shared_data[tid] = max_val;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }

    output[batch_idx] = shared_data[0];
}





torch::Tensor find_max(torch::Tensor input)
{
    int batch_size = input.size(0);
    int dim = input.size(1);

    torch::Tensor output = torch::empty({batch_size, 1}, input.options());

    int threads = 256;
    int blocks = batch_size;
    int shared_mem = threads * sizeof(float);

    find_max_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );

    return output;
}