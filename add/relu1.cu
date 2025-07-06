#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Optimized vectorized kernel - process 4 float4s per thread
__global__ void relu_kernel_vec4(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int base_idx = tid * 4; // 4 float4s = 16 floats per thread
    
    // Unroll loop for better ILP
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int vec_idx = base_idx + i;
        if (vec_idx * 4 < N) {
            float4 v = __ldg(reinterpret_cast<const float4*>(input) + vec_idx);
            v.x = fmaxf(v.x, 0.0f);
            v.y = fmaxf(v.y, 0.0f);
            v.z = fmaxf(v.z, 0.0f);
            v.w = fmaxf(v.w, 0.0f);
            *(reinterpret_cast<float4*>(output) + vec_idx) = v;
        }
    }
}

// Remove separate tail kernel - handle in main kernel
__global__ void relu_kernel_tail(
    const float* __restrict__ input,
    float* __restrict__ output,
    int start,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = start + tid;
    if (i < N) {
        output[i] = fmaxf(__ldg(input + i), 0.0f);
    }
}

torch::Tensor relu(torch::Tensor input, bool in_place = false) {
    TORCH_CHECK(input.is_cuda(), "must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "must be contiguous");
    
    const int N = input.numel();
    auto output = in_place ? input : torch::empty_like(input);
    
    if (N == 0) return output;
    
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();
    
    // Get device properties
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Optimize for memory bandwidth - each thread processes 16 floats
    const int elements_per_thread = 16; // 4 float4s
    const int total_vec4s = N / 4;
    const int vec4s_per_thread = elements_per_thread / 4;
    const int required_threads = (total_vec4s + vec4s_per_thread - 1) / vec4s_per_thread;
    
    // Use high occupancy
    int threads = 256; // Good for most GPUs
    int blocks = min(65535, (required_threads + threads - 1) / threads);
    
    // Ensure we have enough blocks to saturate GPU
    blocks = max(blocks, prop.multiProcessorCount * 2);
    
    // Launch main kernel
    relu_kernel_vec4<<<blocks, threads>>>(in_ptr, out_ptr, N);
    
    // Handle tail elements
    const int processed = (total_vec4s / vec4s_per_thread) * vec4s_per_thread * 4;
    if (processed < N) {
        int tail_threads = 256;
        int tail_blocks = ((N - processed) + tail_threads - 1) / tail_threads;
        relu_kernel_tail<<<tail_blocks, tail_threads>>>(
            in_ptr, out_ptr, processed, N
        );
    }
    
    // Remove sync for better pipeline
    return output;
}