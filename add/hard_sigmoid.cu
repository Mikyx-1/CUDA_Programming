#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__
void hard_sigmoid_kernel(const float* input, float* output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        float curr_val = input[idx];
        float out;
        if (curr_val <= -3)  out = 0.0f;
        else if (curr_val >= 3)
        {
            out = 1.0;
        }
        else{
            out = curr_val / 6 + 0.5f;
        }
        output[idx] = out;
    }
}


torch::Tensor hard_sigmoid(torch::Tensor input, bool in_place = false)
{
    int N = input.numel();

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    if (in_place)
    {
        hard_sigmoid_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            input.data_ptr<float>(),
            N
        );

        return input;
    }
    else{
        torch::Tensor output = torch::empty_like(input);
        hard_sigmoid_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            N
        );

        return output;
    }
}