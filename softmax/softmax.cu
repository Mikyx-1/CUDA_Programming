#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__
void softmax_kernel(const float* input, float* output, int batch_size, int dim)
{
    int batch_idx = threadIdx.x;

    if (batch_idx < batch_size)
    {
        float sum_exp = 0.0f;
        for (int i = 0; i < dim; i++)
        {
            float val = __expf(input[batch_idx * dim + i]);
            output[batch_idx * dim + i] = val;
            sum_exp += val;
        }

        for (int i = 0; i < dim; i++)
        {
            output[batch_idx * dim + i] /= sum_exp;
        }
    }
}



torch::Tensor softmax(torch::Tensor input)
{

    int batch_size = input.size(0);
    int dim = input.size(1);
    torch::Tensor output = torch::empty_like(input);

    dim3 threads(batch_size);
    dim3 blocks(1);

    softmax_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );

    return output;
}