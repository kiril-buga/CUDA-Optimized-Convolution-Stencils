#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Timer class using CUDA events
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }

    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }

    float elapsed_ms() const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

// Block size configurations for benchmarking
struct BlockConfig {
    int x, y;
};

constexpr BlockConfig BLOCK_CONFIGS[] = {
    {8, 8},
    {16, 16},
    {32, 8},
    {32, 16}
};
constexpr int NUM_BLOCK_CONFIGS = sizeof(BLOCK_CONFIGS) / sizeof(BLOCK_CONFIGS[0]);

// V1: Naive global memory convolution
// Each thread computes one output pixel
void conv2d_cuda_v1(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    const float* d_kernel,
    int kernel_size,
    int block_x = 16,
    int block_y = 16
);

// V2: Shared memory tiled convolution
// Uses shared memory for tile + halo
void conv2d_cuda_v2(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    const float* d_kernel,
    int kernel_size,
    int block_x = 16,
    int block_y = 16
);

// Multi-GPU wrapper
// Splits image by rows, processes on multiple GPUs, stitches result
void conv2d_multi_gpu(
    const float* h_input,
    float* h_output,
    int width,
    int height,
    const float* h_kernel,
    int kernel_size,
    int num_gpus = -1  // -1 = use all available
);

// Utility: Copy kernel to constant memory (for V2 optimization)
void set_constant_kernel(const float* kernel, int kernel_size);

// Get device info
void print_device_info();
int get_num_gpus();
