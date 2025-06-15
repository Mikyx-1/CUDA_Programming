#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void convolution_kernel(const float* input, const float* kernel, float* output, 
                        int H, int W, int KH, int KW)
{

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y >= H || x >= W) return ;

    int pad_h = KH / 2;
    int pad_w = KW / 2;

    float acc = 0.0f;

    for (int i = 0; i < KW; i++)
    {
        for (int j = 0; j < KH; j++)
        {
            int input_y = y + j - pad_h;
            int input_x = x + i - pad_w;

            if (input_y >= 0 && input_y < H && input_x >= 0 && input_x < W)
            {
            acc += input[input_y * W + input_x] * kernel[j * KW + i];
            }
        }
    }
    output[y * W + x] = acc;
}



torch::Tensor convol2D(torch::Tensor input, torch::Tensor kernel)
{
    // Input sizes
    int H = input.size(0);
    int W = input.size(1);
    int KH = kernel.size(0);
    int KW = kernel.size(1);

    torch::Tensor output = torch::empty({H, W}, input.options());

    // Launch parameters
    dim3 threads(KW, KH);
    dim3 blocks((W + threads.x - 1) / threads.x,
                (H + threads.y - 1) / threads.y);

    // Launch kernel
    convolution_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        H, 
        W, 
        KH,
        KW
    );

    return output;
}
