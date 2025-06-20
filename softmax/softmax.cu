#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void softmax_kernel(const float* input, float* output, int batch_size, int dim) {
    extern __shared__ float shared_data[];

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    const float* input_row = input + batch_idx * dim;
    float* output_row = output + batch_idx * dim;

    // Step 1: find max for numerical stability
    float max_val = -INFINITY;
    for (int i = tid; i < dim; i += blockDim.x) {
        max_val = fmaxf(max_val, input_row[i]);
    }
    // Reduction to get global max
    shared_data[tid] = max_val;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        __syncthreads();
    }
    float row_max = shared_data[0];

    // Step 2: compute exponentials and sum
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = expf(input_row[i] - row_max);
        output_row[i] = val;
        sum += val;
    }

    // Reduction to get total sum
    shared_data[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared_data[tid] += shared_data[tid + stride];
        __syncthreads();
    }
    float row_sum = shared_data[0];

    // Step 3: normalize
    for (int i = tid; i < dim; i += blockDim.x) {
        output_row[i] /= row_sum;
    }
}


torch::Tensor softmax(torch::Tensor input)
{
    int batch_size = input.size(0);
    int dim = input.size(1);

    torch::Tensor output = torch::empty_like(input);

    int threads = 256;      // Must be the power of 2 (2^n)
    int shared_mem = threads * sizeof(float);

    softmax_kernel<<<batch_size, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );

    return output;
}