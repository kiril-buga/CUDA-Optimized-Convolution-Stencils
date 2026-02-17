#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>

#include "cpu_convolution.h"
#include "filters.h"
#include "image_utils.h"

#ifndef CPU_ONLY
#include "cuda_convolution.cuh"
#endif

using Clock = std::chrono::high_resolution_clock;

// CPU timing helper
double time_cpu_convolution(
    const std::vector<float>& input,
    std::vector<float>& output,
    int width,
    int height,
    const std::vector<float>& kernel,
    int kernel_size,
    int iterations = 3
) {
    double total_ms = 0.0;
    for (int i = 0; i < iterations; ++i) {
        auto start = Clock::now();
        conv2d_cpu(input, output, width, height, kernel, kernel_size);
        auto end = Clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }
    return total_ms / iterations;
}

#ifndef CPU_ONLY

// GPU timing helper — V1
double time_gpu_convolution_v1(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    const float* d_kernel,
    int kernel_size,
    int block_x = 16,
    int block_y = 16,
    int iterations = 10
) {
    // Warmup
    conv2d_cuda_v1(d_input, d_output, width, height, d_kernel, kernel_size, block_x, block_y);
    cudaDeviceSynchronize();

    CudaTimer timer;
    double total_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        timer.start();
        conv2d_cuda_v1(d_input, d_output, width, height, d_kernel, kernel_size, block_x, block_y);
        timer.stop();
        total_ms += timer.elapsed_ms();
    }

    return total_ms / iterations;
}

// GPU timing helper — V1_const
double time_gpu_convolution_v1_const(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    const float* d_kernel,
    int kernel_size,
    int block_x = 16,
    int block_y = 16,
    int iterations = 10
) {
    // Warmup
    conv2d_cuda_v1_const(d_input, d_output, width, height, d_kernel, kernel_size, block_x, block_y);
    cudaDeviceSynchronize();

    CudaTimer timer;
    double total_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        timer.start();
        conv2d_cuda_v1_const(d_input, d_output, width, height, d_kernel, kernel_size, block_x, block_y);
        timer.stop();
        total_ms += timer.elapsed_ms();
    }

    return total_ms / iterations;
}

// GPU timing helper — V2
double time_gpu_convolution_v2(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    const float* d_kernel,
    int kernel_size,
    int block_x = 16,
    int block_y = 16,
    int iterations = 10
) {
    // Warmup
    conv2d_cuda_v2(d_input, d_output, width, height, d_kernel, kernel_size, block_x, block_y);
    cudaDeviceSynchronize();

    CudaTimer timer;
    double total_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        timer.start();
        conv2d_cuda_v2(d_input, d_output, width, height, d_kernel, kernel_size, block_x, block_y);
        timer.stop();
        total_ms += timer.elapsed_ms();
    }

    return total_ms / iterations;
}

#endif // CPU_ONLY

void print_separator(const std::string& title = "") {
    std::cout << "\n" << std::string(70, '=') << "\n";
    if (!title.empty()) {
        std::cout << "  " << title << "\n";
        std::cout << std::string(70, '=') << "\n";
    }
}

void run_correctness_tests() {
    print_separator("CORRECTNESS TESTS");

    const int width = 64;
    const int height = 64;

    auto input = image_utils::checkerboard(width, height, 8);
    std::vector<float> cpu_output, gpu_output;

    std::vector<std::pair<std::string, int>> test_cases = {
        {"gaussian", 3},
        {"gaussian", 5},
        {"gaussian", 7},
        {"sobel_x", 3},
        {"laplacian", 3},
        {"box_blur", 3},
        {"box_blur", 5},
    };

    for (const auto& [filter_name, kernel_size] : test_cases) {
        auto kernel = filters::get_kernel(filter_name, kernel_size);

        // CPU reference
        conv2d_cpu(input, cpu_output, width, height, kernel, kernel_size);

#ifndef CPU_ONLY
        // GPU
        float *d_input, *d_output, *d_kernel;
        CUDA_CHECK(cudaMalloc(&d_input, width * height * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, width * height * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_input, input.data(), width * height * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data(), kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

        // Test V1
        conv2d_cuda_v1(d_input, d_output, width, height, d_kernel, kernel_size);
        gpu_output.resize(width * height);
        CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost));
        float v1_error = image_utils::max_abs_error(cpu_output, gpu_output);

        // Test V1_const
        conv2d_cuda_v1_const(d_input, d_output, width, height, d_kernel, kernel_size);
        CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost));
        float v1c_error = image_utils::max_abs_error(cpu_output, gpu_output);

        // Test V2
        conv2d_cuda_v2(d_input, d_output, width, height, d_kernel, kernel_size);
        CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost));
        float v2_error = image_utils::max_abs_error(cpu_output, gpu_output);

        std::cout << std::setw(12) << filter_name << " " << kernel_size << "x" << kernel_size
                  << "  V1: " << std::scientific << std::setprecision(2) << v1_error
                  << "  V1c: " << v1c_error
                  << "  V2: " << v2_error
                  << (v1_error < 1e-5 && v1c_error < 1e-5 && v2_error < 1e-5 ? "  [PASS]" : "  [FAIL]")
                  << std::fixed << "\n";

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_kernel));
#else
        std::cout << std::setw(12) << filter_name << " " << kernel_size << "x" << kernel_size
                  << "  CPU OK\n";
#endif
    }

}

void run_benchmark_image_sizes() {
    print_separator("BENCHMARK: Image Size Scaling (Gaussian 3x3)");

    std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
    auto kernel = filters::gaussian_3x3();
    const int kernel_size = 3;

    std::cout << std::setw(10) << "Size"
              << std::setw(12) << "CPU (ms)"
#ifndef CPU_ONLY
              << std::setw(12) << "V1 (ms)"
              << std::setw(12) << "V1c (ms)"
              << std::setw(12) << "V2 (ms)"
              << std::setw(12) << "Speedup V1"
              << std::setw(12) << "Speedup V2"
#endif
              << "\n";
    std::cout << std::string(82, '-') << "\n";

    for (int size : sizes) {
        auto input = image_utils::random_noise(size, size);
        std::vector<float> output;

        double cpu_time = time_cpu_convolution(input, output, size, size, kernel, kernel_size);

#ifndef CPU_ONLY
        float *d_input, *d_output, *d_kernel;
        CUDA_CHECK(cudaMalloc(&d_input, size * size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, size * size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_input, input.data(), size * size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data(), kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

        double v1_time = time_gpu_convolution_v1(d_input, d_output, size, size, d_kernel, kernel_size);
        double v1c_time = time_gpu_convolution_v1_const(d_input, d_output, size, size, d_kernel, kernel_size);
        double v2_time = time_gpu_convolution_v2(d_input, d_output, size, size, d_kernel, kernel_size);

        std::cout << std::setw(10) << (std::to_string(size) + "x" + std::to_string(size))
                  << std::setw(12) << std::fixed << std::setprecision(3) << cpu_time
                  << std::setw(12) << v1_time
                  << std::setw(12) << v1c_time
                  << std::setw(12) << v2_time
                  << std::setw(12) << std::setprecision(1) << (cpu_time / v1_time) << "x"
                  << std::setw(12) << (cpu_time / v2_time) << "x"
                  << "\n";

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_kernel));
#else
        std::cout << std::setw(10) << (std::to_string(size) + "x" + std::to_string(size))
                  << std::setw(12) << std::fixed << std::setprecision(3) << cpu_time
                  << "\n";
#endif
    }
}

void run_benchmark_kernel_sizes() {
    print_separator("BENCHMARK: Kernel Sizes (1024x1024 image)");

    const int size = 1024;
    std::vector<int> kernel_sizes = {3, 5, 7};

    auto input = image_utils::random_noise(size, size);
    std::vector<float> output;

    std::cout << std::setw(10) << "Kernel"
              << std::setw(12) << "CPU (ms)"
#ifndef CPU_ONLY
              << std::setw(12) << "V1 (ms)"
              << std::setw(12) << "V1c (ms)"
              << std::setw(12) << "V2 (ms)"
              << std::setw(10) << "V2/V1"
#endif
              << "\n";
    std::cout << std::string(68, '-') << "\n";

    for (int ks : kernel_sizes) {
        auto kernel = filters::get_kernel("gaussian", ks);

        double cpu_time = time_cpu_convolution(input, output, size, size, kernel, ks);

#ifndef CPU_ONLY
        float *d_input, *d_output, *d_kernel;
        CUDA_CHECK(cudaMalloc(&d_input, size * size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, size * size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_kernel, ks * ks * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_input, input.data(), size * size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data(), ks * ks * sizeof(float), cudaMemcpyHostToDevice));

        double v1_time = time_gpu_convolution_v1(d_input, d_output, size, size, d_kernel, ks);
        double v1c_time = time_gpu_convolution_v1_const(d_input, d_output, size, size, d_kernel, ks);
        double v2_time = time_gpu_convolution_v2(d_input, d_output, size, size, d_kernel, ks);

        std::cout << std::setw(10) << (std::to_string(ks) + "x" + std::to_string(ks))
                  << std::setw(12) << std::fixed << std::setprecision(3) << cpu_time
                  << std::setw(12) << v1_time
                  << std::setw(12) << v1c_time
                  << std::setw(12) << v2_time
                  << std::setw(10) << std::setprecision(2) << (v1_time / v2_time) << "x"
                  << "\n";

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_kernel));
#else
        std::cout << std::setw(10) << (std::to_string(ks) + "x" + std::to_string(ks))
                  << std::setw(12) << std::fixed << std::setprecision(3) << cpu_time
                  << "\n";
#endif
    }
}

#ifndef CPU_ONLY

void run_benchmark_block_sizes() {
    print_separator("BENCHMARK: Block Size Sweep (2048x2048, Gaussian 5x5)");

    const int size = 2048;
    const int kernel_size = 5;
    auto kernel = filters::get_kernel("gaussian", kernel_size);
    auto input = image_utils::random_noise(size, size);

    float *d_input, *d_output, *d_kernel;
    CUDA_CHECK(cudaMalloc(&d_input, size * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, size * size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, input.data(), size * size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data(), kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    struct BlockConfig { int x, y; };
    std::vector<BlockConfig> configs = {{8,8}, {16,16}, {32,8}, {32,16}, {32,32}};

    std::cout << std::setw(10) << "Block"
              << std::setw(10) << "Threads"
              << std::setw(12) << "V1 (ms)"
              << std::setw(12) << "V1c (ms)"
              << std::setw(12) << "V2 (ms)"
              << std::setw(10) << "V2/V1"
              << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (const auto& cfg : configs) {
        double v1_time = time_gpu_convolution_v1(d_input, d_output, size, size, d_kernel, kernel_size,
                                                  cfg.x, cfg.y);
        double v1c_time = time_gpu_convolution_v1_const(d_input, d_output, size, size, d_kernel, kernel_size,
                                                         cfg.x, cfg.y);
        double v2_time = time_gpu_convolution_v2(d_input, d_output, size, size, d_kernel, kernel_size,
                                                  cfg.x, cfg.y);

        std::string block_str = std::to_string(cfg.x) + "x" + std::to_string(cfg.y);
        std::cout << std::setw(10) << block_str
                  << std::setw(10) << (cfg.x * cfg.y)
                  << std::setw(12) << std::fixed << std::setprecision(3) << v1_time
                  << std::setw(12) << v1c_time
                  << std::setw(12) << v2_time
                  << std::setw(10) << std::setprecision(2) << (v1_time / v2_time) << "x"
                  << "\n";
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_kernel));
}

#endif // CPU_ONLY

int main(int argc, char** argv) {
    std::cout << "CUDA Convolution Benchmark\n";
    std::cout << "==========================\n\n";

#ifndef CPU_ONLY
    print_device_info();
#else
    std::cout << "Built in CPU-only mode\n";
#endif

    // Run all tests and benchmarks
    run_correctness_tests();
    run_benchmark_image_sizes();
    run_benchmark_kernel_sizes();

#ifndef CPU_ONLY
    run_benchmark_block_sizes();
#endif

    print_separator("BENCHMARK COMPLETE");

    return 0;
}
