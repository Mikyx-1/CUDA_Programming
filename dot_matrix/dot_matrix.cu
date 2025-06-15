#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__
void dot_matrix_kernel(const float* a, const float* b, float* c, int a_height, int a_width, int b_width, int b_height)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < a_height && x < b_width)
    {
        float sum = 0.0f;
        for (int i = 0; i < a_width; i++)
        {
            sum += a[y * a_width + i] * b[i * b_width + x];
        }

        c[y * b_width + x] = sum;
    }
}


torch::Tensor dot_matrix(torch::Tensor a, torch::Tensor b)
{
    int a_height = a.size(0);
    int a_width = a.size(1);

    int b_height = b.size(0); 
    int b_width = b.size(1);

    auto c = torch::zeros({a_height, b_width}, a.options());

    dim3 threads(16, 16);
    dim3 blocks((a_height + threads.x - 1) / threads.x, 
                (b_width + threads.y - 1) / threads.y);

    dot_matrix_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        a_height,
        a_width,
        b_width,
        b_height
    );

    return c;
}
