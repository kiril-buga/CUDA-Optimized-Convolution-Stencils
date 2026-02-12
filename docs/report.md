# CUDA-Optimized 2D Convolution: A Study in GPU Parallel Computing

**Course:** Parallel and Distributed Computing
**Author:** [TODO: Name]
**Date:** [TODO: Date]
**Institution:** [TODO: University]

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background: Parallel Computing](#2-background-parallel-computing)
3. [Background: Image Processing](#3-background-image-processing)
4. [Project Methodology and Implementation](#4-project-methodology-and-implementation)
5. [Results](#5-results)
6. [Conclusion](#6-conclusion)
7. [References](#7-references)

---

## 1. Introduction

### 1.1 Motivation

Modern image processing and computer vision workloads involve applying the same mathematical operation to millions of pixels. A 4K image contains over 8 million pixels, and even a simple 3x3 filter requires 9 multiply-accumulate operations per pixel, totaling over 75 million floating-point operations for a single pass. On a conventional CPU, these operations execute sequentially, making large-scale image processing slow and impractical for real-time applications.

Graphics Processing Units (GPUs) offer a fundamentally different computing model. Where a CPU optimizes for single-thread latency with a handful of powerful cores, a GPU provides thousands of lightweight cores optimized for throughput. 2D convolution is an ideal candidate for GPU acceleration because every output pixel can be computed independently, making it an embarrassingly parallel workload. This project explores how to systematically exploit this parallelism using NVIDIA's CUDA platform, progressing from a naive implementation to increasingly optimized versions that leverage the GPU's memory hierarchy and execution model.

### 1.2 Objectives

- Implement a correct sequential CPU baseline for 2D convolution with zero-padding boundary handling.
- Develop three progressively optimized CUDA kernels: a naive global memory version (V1), a shared memory tiled version (V2), and a fused multi-operation kernel (V3).
- Implement a multi-GPU framework that distributes work across multiple devices using domain decomposition.
- Benchmark performance across four dimensions: image size, filter kernel size, thread block configuration, and kernel fusion strategy.
- Verify the correctness of all GPU implementations against the CPU reference using quantitative error metrics.

### 1.3 Report Organization

Section 2 introduces the parallel computing concepts that underpin GPU programming, covering architecture, memory hierarchy, and optimization techniques. Section 3 provides a concise overview of the image processing operations used as the computational workload. Section 4 describes the project's implementation and benchmarking methodology in detail. Section 5 presents the experimental results, and Section 6 concludes with findings and future work.

---

## 2. Background: Parallel Computing

### 2.1 Why Parallelism?

The traditional CPU architecture is built around a small number of powerful cores (typically 4-16), each equipped with deep pipelines, branch predictors, and large multi-level caches. This design minimizes the latency of individual operations, making CPUs excellent for sequential, control-flow-heavy workloads. However, when the same operation must be applied to millions of independent data elements, the CPU's per-core performance advantage becomes irrelevant -- what matters is throughput.

A GPU inverts this trade-off. Instead of a few complex cores, a GPU contains hundreds or thousands of simpler cores grouped into processing units called Streaming Multiprocessors (SMs). Each core lacks the sophisticated control logic of a CPU core, but the sheer number of cores enables massive data parallelism. The GPU model assumes that if one thread stalls (e.g., waiting for a memory access), another thread is ready to execute immediately, keeping the hardware busy. This is known as latency hiding through occupancy.

Amdahl's Law provides the theoretical framework for understanding parallelization gains. If a fraction *p* of a program is parallelizable and the remaining fraction *(1 - p)* is inherently sequential, the maximum speedup with *N* processors is:

```
Speedup = 1 / ((1 - p) + p/N)
```

For 2D convolution, the parallel fraction is extremely high: every output pixel is computed independently from its local neighborhood in the input image. The only sequential components are memory allocation, data transfer between host (CPU) and device (GPU), and result collection. This makes convolution an ideal workload for GPU acceleration, with theoretical speedups approaching the ratio of GPU-to-CPU computational throughput.

### 2.2 GPU Architecture Fundamentals

#### 2.2.1 Streaming Multiprocessors and CUDA Cores

An NVIDIA GPU is organized as an array of Streaming Multiprocessors (SMs). Each SM is an independent processing unit containing multiple CUDA cores (also called shader processors), a register file, shared memory, and scheduling logic. The GPU used in this project, the NVIDIA GeForce GTX 1050 Ti (Pascal architecture, compute capability 6.1), has 6 SMs with 128 CUDA cores each, for a total of 768 CUDA cores.

When a CUDA kernel is launched, the GPU scheduler distributes thread blocks across the available SMs. Each thread block runs entirely on a single SM and cannot migrate to another. An SM can host multiple thread blocks concurrently, limited by the SM's resources (registers, shared memory, and maximum thread count). This block-to-SM mapping is the fundamental unit of work distribution on the GPU.

#### 2.2.2 Thread Hierarchy: Threads, Warps, Blocks, and Grids

CUDA organizes parallel execution in a four-level hierarchy:

- **Thread:** The smallest unit of execution. Each thread has a unique ID within its block (`threadIdx.x`, `threadIdx.y`) and runs the same kernel code on different data. In this project, each thread typically computes one output pixel.

- **Warp:** A group of 32 threads that execute in lockstep using Single Instruction, Multiple Threads (SIMT) execution. All threads in a warp execute the same instruction at the same time. If threads in a warp take different branches (warp divergence), both paths are serialized, reducing efficiency. The warp is the fundamental scheduling unit on the GPU.

- **Thread Block:** A programmer-defined grouping of threads (e.g., 16x16 = 256 threads, or 32x8 = 256 threads). Threads within a block can cooperate through shared memory and synchronize with barriers. Blocks are assigned to SMs and execute independently of other blocks.

- **Grid:** The collection of all thread blocks needed to process the entire input. For a 1024x1024 image with 16x16 thread blocks, the grid is 64x64 = 4,096 blocks, each containing 256 threads, for a total of over one million threads.

In the project's code, the grid and block dimensions are computed as:

```cuda
dim3 block(block_x, block_y);
dim3 grid((width + block_x - 1) / block_x,
          (height + block_y - 1) / block_y);
```

This ceiling-division formula ensures the grid covers the entire image, even when dimensions are not evenly divisible by the block size. Threads that map to positions outside the image boundaries are deactivated with a bounds check.

#### 2.2.3 Memory Hierarchy

The GPU memory hierarchy is central to understanding performance optimization. Each level trades capacity for speed:

- **Registers:** The fastest storage, private to each thread. Variables like loop counters and accumulators reside in registers. Access latency is approximately 1 clock cycle. The GTX 1050 Ti provides 65,536 32-bit registers per SM.

- **Shared Memory:** A fast, programmer-managed memory space shared by all threads in a block. It acts as a software-controlled cache. Access latency is roughly 5-10 cycles, making it approximately 100x faster than global memory. The GTX 1050 Ti provides 48 KB of shared memory per SM. This is the key resource exploited by the V2 tiled convolution kernel.

- **Constant Memory:** A 64 KB read-only memory space, cached on each SM. When all threads in a warp read the same address, the value is broadcast to all threads in a single transaction. This makes it ideal for small, read-only data accessed uniformly, such as convolution filter coefficients.

- **Global Memory:** The largest memory space (4 GB on the GTX 1050 Ti), accessible by all threads across all SMs. It has the highest latency (400-600 clock cycles) and a peak bandwidth of 112 GB/s. The input and output image arrays reside in global memory. Efficient use of global memory requires coalesced access patterns (see Section 2.3.1).

### 2.3 Key Optimization Techniques

#### 2.3.1 Memory Coalescing

When threads in a warp access contiguous addresses in global memory, the hardware combines these individual requests into a single wide memory transaction (typically 128 bytes). This is called memory coalescing and is critical for achieving high bandwidth utilization. Conversely, scattered or strided access patterns result in multiple separate transactions, wasting bandwidth.

In 2D image processing, images are stored in row-major order: `pixel = image[y * width + x]`. When the block's x-dimension equals the warp size (32), threads in the same warp process adjacent pixels in the same row, resulting in perfectly coalesced reads. This is why the benchmarks test a `32x8` block configuration, which has the same total thread count (256) as `16x16` but aligns each row of threads with a full warp.

#### 2.3.2 Tiling and Shared Memory

The naive convolution kernel (V1) has a fundamental inefficiency: each thread reads its pixel's entire neighborhood from global memory independently. For a 3x3 kernel, neighboring threads share 6 of their 9 input pixels, but each thread fetches all 9 from global memory. For larger kernels, this redundancy grows rapidly.

Shared memory tiling solves this problem. The idea is to cooperatively load a tile of input data, including a boundary region called the halo, into shared memory once. Then all threads in the block compute their output pixels by reading from the fast shared memory instead of slow global memory.

The tile dimensions include the halo:

```
SHARED_W = TILE_W + 2 * HALF_K
SHARED_H = TILE_H + 2 * HALF_K
```

where `TILE_W x TILE_H` is the output region (matching the block dimensions) and `HALF_K = kernel_size / 2` is the halo width on each side. For a 16x16 block with a 3x3 kernel, the shared memory tile is 18x18 = 324 floats. Each of the 256 threads in the block reads 9 values from shared memory during convolution, yielding a data reuse factor of (256 x 9) / 324 = 7.1x. This means each value loaded from global memory is used approximately 7 times, dramatically reducing bandwidth pressure.

The loading phase may require each thread to load more than one element, since the shared memory tile (e.g., 18x18 = 324 elements) is larger than the number of threads in the block (e.g., 256). The code handles this with a nested loop:

```cuda
const int num_loads_x = (SHARED_W + BLOCK_X - 1) / BLOCK_X;
const int num_loads_y = (SHARED_H + BLOCK_Y - 1) / BLOCK_Y;

for (int ly = 0; ly < num_loads_y; ++ly) {
    for (int lx = 0; lx < num_loads_x; ++lx) {
        // Each thread loads tile[shared_y][shared_x] from global memory
    }
}
```

Pixels outside the image boundary are loaded as zero, implementing zero-padding implicitly.

#### 2.3.3 Constant Memory

The convolution filter kernel is a small array (9 to 49 floats for 3x3 through 7x7 kernels), is read-only during execution, and is accessed with the same index pattern by all threads. These properties make it a textbook use case for constant memory.

In the project, the filter is stored in a `__constant__` array:

```cuda
__constant__ float c_kernel[MAX_KERNEL_ELEMENTS];
```

and copied from host memory using `cudaMemcpyToSymbol()`. During convolution, all threads in a warp access `c_kernel[ky * KERNEL_SIZE + kx]` with the same indices at each step, triggering a single cached read that is broadcast to all 32 threads. This eliminates 31 redundant global memory reads per warp per kernel element.

#### 2.3.4 Kernel Fusion

A common image processing pipeline applies multiple filters in sequence. For example, edge detection often requires Gaussian smoothing followed by Sobel gradient computation. Without fusion, this pipeline requires three separate kernel launches (Gaussian, Sobel X, Sobel Y), each writing intermediate results to global memory and reading them back.

Kernel fusion combines multiple operations into a single kernel launch, eliminating intermediate global memory traffic. The fused kernel computes the Gaussian blur for each pixel in the Sobel neighborhood, applies the Sobel operators to the blurred values, and writes only the final edge magnitude to global memory.

However, fusion introduces a trade-off. The fused kernel in this project uses a simple per-thread approach without shared memory tiling. Each thread must compute 9 Gaussian blur values (one for each position in the 3x3 Sobel neighborhood), each requiring 9 global memory reads, totaling approximately 81 global reads per output pixel. The unfused path uses the optimized V2 kernel with shared memory tiling, which achieves far fewer global reads per pixel through data reuse. This demonstrates that fusion alone is not sufficient for performance gains -- it must be combined with memory access optimizations.

#### 2.3.5 Template Specialization and Loop Unrolling

The V2 shared memory kernel is implemented as a C++ template parameterized on `BLOCK_X`, `BLOCK_Y`, and `KERNEL_SIZE`:

```cuda
template <int BLOCK_X, int BLOCK_Y, int KERNEL_SIZE>
__global__ void conv2d_kernel_v2_shared(...)
```

This allows the compiler to treat tile dimensions, halo sizes, and loop bounds as compile-time constants. The shared memory allocation `__shared__ float tile[SHARED_H][SHARED_W]` uses statically known dimensions, and the convolution loops can be fully unrolled with `#pragma unroll`:

```cuda
#pragma unroll
for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
    #pragma unroll
    for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
        acc += tile[sy][sx] * c_kernel[ky * KERNEL_SIZE + kx];
    }
}
```

Loop unrolling eliminates branch instructions and loop overhead, replacing them with a straight-line sequence of multiply-accumulate operations. For a 3x3 kernel, this produces 9 inline operations with no loop control overhead. The trade-off is increased binary size: the project instantiates 12 template variants (4 block sizes x 3 kernel sizes), and a runtime dispatch macro selects the correct one.

#### 2.3.6 Synchronization Barriers

When threads cooperatively load data into shared memory, all threads must finish loading before any thread begins reading. Without synchronization, a thread could read a shared memory location that has not yet been written by another thread, leading to incorrect results.

CUDA provides the `__syncthreads()` intrinsic, which acts as a block-level barrier: execution halts until every thread in the block has reached the barrier. In the V2 kernel, `__syncthreads()` is placed between the loading phase and the computation phase:

```cuda
// Phase 1: All threads cooperatively load tile into shared memory
tile[shared_y][shared_x] = input[global_y * width + global_x];

__syncthreads();  // Barrier: wait for all loads to complete

// Phase 2: Each thread computes convolution from shared memory
acc += tile[sy][sx] * c_kernel[ky * KERNEL_SIZE + kx];
```

This barrier is block-scoped -- it does not synchronize across different blocks. Cross-block synchronization requires separate kernel launches or atomic operations.

### 2.4 Multi-GPU Computing

#### 2.4.1 Domain Decomposition

When a single GPU's capacity is insufficient or when multiple GPUs are available, the workload can be distributed across devices. This project implements 1D domain decomposition along the y-axis: the image is split into horizontal strips, one per GPU. Each GPU processes `height / num_gpus` rows (with remaining rows distributed to the first GPUs):

```cpp
const int base_rows = height / num_gpus;
int remaining = height % num_gpus;
int rows_for_this_gpu = base_rows + (g < remaining ? 1 : 0);
```

This strategy keeps the implementation simple and ensures approximately equal work distribution. Each strip is processed independently using the V2 kernel.

#### 2.4.2 Halo Regions and Overlap

Because convolution reads neighboring pixels, each GPU's strip must include extra rows beyond its assigned region. For a kernel of size K, each boundary needs K/2 additional rows from the adjacent strip. These extra rows form the halo region:

```cpp
int actual_start = std::max(0, start_rows[g] - half_k);
int actual_end   = std::min(height, end_rows[g] + half_k);
```

The first GPU has no upper halo (it starts at row 0), and the last GPU has no lower halo (it ends at the last row). Interior GPUs have halos on both sides. After computation, only the originally assigned rows (excluding halos) are copied back to the host, avoiding duplicate data in the final output.

#### 2.4.3 CUDA Streams and Asynchronous Transfers

Each GPU operates with its own CUDA stream, an ordered queue of operations (memory transfers and kernel launches) that execute sequentially within the stream but can overlap with operations in other streams. This enables concurrent execution across GPUs:

```cpp
cudaStreamCreate(&streams[g]);
cudaMemcpyAsync(d_inputs[g], h_input + actual_start * width,
                chunk_size, cudaMemcpyHostToDevice, streams[g]);
```

The `cudaMemcpyAsync` function returns immediately, allowing the host to enqueue work on the next GPU without waiting for the transfer to complete. After all work is submitted, `cudaStreamSynchronize` blocks until each GPU finishes. This overlap maximizes hardware utilization: while one GPU computes, another can be transferring data.

---

## 3. Background: Image Processing

### 3.1 2D Convolution

2D convolution is a mathematical operation that combines an input image *I* with a small matrix called a kernel (or filter) *K* to produce an output image *O*. For each pixel in the output, the kernel is centered on the corresponding input pixel, element-wise products are computed between the kernel and the overlapping image region, and the results are summed:

```
O(x, y) = sum over (kx, ky) of I(x + kx, y + ky) * K(kx, ky)
```

The kernel slides across every pixel position in the image. Different kernel values produce different effects: smoothing, edge detection, sharpening, and more. All images in this project are single-channel (grayscale), stored as 1D arrays of `float` values in the range [0, 1] using row-major order (`pixel = image[y * width + x]`).

### 3.2 Zero-Padding Boundary Handling

When the kernel overlaps the edge of the image, some kernel positions fall outside the image boundary. Zero-padding treats these out-of-bounds pixels as having a value of zero. In the implementation, this is handled with bounds checking:

```cpp
if (ix < 0 || ix >= width) continue;   // skip: contributes 0
if (iy < 0 || iy >= height) continue;
```

This is the simplest boundary handling strategy and preserves the output image dimensions (the output is the same size as the input).

### 3.3 Common Filters

The project implements several standard convolution filters:

| Filter | Sizes | Purpose |
|--------|-------|---------|
| Gaussian | 3x3, 5x5, 7x7 | Smoothing / noise reduction (weighted average, center-heavy) |
| Sobel X / Y | 3x3 | Edge detection (horizontal / vertical gradient approximation) |
| Laplacian | 3x3 | Edge detection (second-order derivative, isotropic) |
| Box Blur | 3x3, 5x5 | Uniform smoothing (equal weights) |
| Sharpen | 3x3 | Edge enhancement |
| Emboss | 3x3 | Relief / emboss visual effect |
| Identity | any odd | Pass-through (no change), used for correctness testing |

As an example, the 3x3 Gaussian kernel weights the center pixel most heavily and gradually decreases influence outward:

```
1/16   2/16   1/16
2/16   4/16   2/16
1/16   2/16   1/16
```

The Sobel X kernel detects vertical edges by computing horizontal intensity differences:

```
-1   0   1
-2   0   2
-1   0   1
```

### 3.4 Why Convolution is a Good Parallel Workload

2D convolution is well suited for GPU parallelization for three reasons:

1. **Data independence:** Each output pixel depends only on a small local neighborhood in the input image. There are no data dependencies between output pixels, making the operation embarrassingly parallel.

2. **Regular access patterns:** The kernel slides across the image in a predictable pattern, producing regular, structured memory accesses that are amenable to coalescing and tiling.

3. **Scalable arithmetic intensity:** The number of operations per pixel scales as O(K^2) with kernel size K. Larger kernels increase the computation-to-memory ratio, shifting the workload from memory-bound toward compute-bound, which is favorable for GPU execution.

---

## 4. Project Methodology and Implementation

### 4.1 Project Structure and Build System

The project is organized into modular source files with clear separation of concerns:

```
src/
  main.cpp                  Benchmark harness, correctness tests, timing
  cpu_convolution.cpp / .h  Sequential CPU reference implementation
  cuda_convolution.cu / .cuh GPU kernels (V1, V2, V3) and multi-GPU logic
  filters.cpp / .h          Filter coefficient definitions
  image_utils.h             Synthetic image generators and error metrics
Makefile                    Build configuration
```

The Makefile supports two build modes:

- **Full CUDA build** (`make all`): compiles both CPU and GPU code using `g++` (C++20) for host code and `nvcc` (C++17) for CUDA device code. The target GPU architecture is configurable via `CUDA_ARCH` (default: 75 for Turing).
- **CPU-only build** (`make cpu_only`): compiles only the CPU code with the `-DCPU_ONLY` preprocessor flag, which conditionally excludes all CUDA code. This allows the project to build and run correctness tests on machines without an NVIDIA GPU.

### 4.2 CPU Baseline

The CPU implementation serves as the correctness reference against which all GPU versions are validated. It uses a straightforward four-level nested loop: for each output pixel (x, y), iterate over all kernel positions (kx, ky), accumulate the product of the input pixel and the corresponding kernel weight, and write the result to the output:

```cpp
for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
        float acc = 0.0f;
        for (int ky = -half_k; ky <= half_k; ++ky) {
            for (int kx = -half_k; kx <= half_k; ++kx) {
                // bounds check + accumulate
                acc += input[iy * width + ix] * kernel[kernel_idx];
            }
        }
        output[y * width + x] = acc;
    }
}
```

The implementation supports arbitrary odd-sized kernels (3x3, 5x5, 7x7) and validates input dimensions at runtime. Out-of-bounds kernel positions are skipped (zero-padding).

### 4.3 GPU V1: Naive Global Memory Kernel

The first GPU implementation maps one thread to each output pixel. Each thread computes the full convolution independently by reading all required input values directly from global memory:

```cuda
__global__ void conv2d_kernel_v1(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width, int height,
    const float* __restrict__ kernel,
    int kernel_size
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float acc = 0.0f;
    for (int ky = -half_k; ky <= half_k; ++ky)
        for (int kx = -half_k; kx <= half_k; ++kx)
            acc += input[(y+ky)*width + (x+kx)] * kernel[...];

    output[y * width + x] = acc;
}
```

The `__restrict__` qualifier tells the compiler that the `input`, `output`, and `kernel` pointers do not alias each other, enabling more aggressive optimizations. Threads that map outside the image bounds exit early.

V1 serves as the baseline for measuring the impact of each subsequent optimization. Its primary limitation is excessive global memory traffic: for a 3x3 kernel, each thread performs 9 global memory reads. Adjacent threads read heavily overlapping neighborhoods, but this overlap is not exploited.

### 4.4 GPU V2: Shared Memory Tiled Kernel

V2 addresses V1's redundant memory accesses through shared memory tiling (described in Section 2.3.2). The algorithm proceeds in two phases:

**Phase 1 -- Cooperative Loading:** All threads in the block collaboratively load the tile (including halo) from global memory into shared memory. Each thread may load multiple elements to cover the shared memory region, which is larger than the block.

**Phase 2 -- Computation:** After synchronizing with `__syncthreads()`, each thread computes its output pixel by reading only from shared memory (for image data) and constant memory (for filter coefficients). The convolution loops are annotated with `#pragma unroll` for compile-time unrolling.

The kernel is templated on `BLOCK_X`, `BLOCK_Y`, and `KERNEL_SIZE`, making tile dimensions compile-time constants. A dispatch macro in the wrapper function selects the correct template instantiation at runtime from 12 pre-compiled variants (4 block sizes x 3 kernel sizes). If the requested configuration is not pre-compiled, the wrapper falls back to V1.

### 4.5 GPU V3: Fused Gaussian + Sobel Kernel

V3 fuses a Gaussian blur followed by Sobel edge detection into a single kernel launch. For each output pixel, the kernel:

1. Computes 9 Gaussian-blurred values (the 3x3 neighborhood around the pixel).
2. Applies the Sobel X and Sobel Y operators to these blurred values.
3. Outputs the edge magnitude as |Gx| + |Gy|.

The three filter kernels (Gaussian 3x3, Sobel X, Sobel Y) are stored in separate constant memory arrays. Intermediate blurred values are held in a per-thread local array (`float blurred[3][3]`) that resides in registers.

The current V3 implementation does not use shared memory tiling. Each thread performs approximately 81 global memory reads (9 Gaussian positions x 9 kernel reads each). This is intentionally left as a simpler version to demonstrate the concept of kernel fusion and the trade-offs involved. As shown in the results, fusion without memory optimization can actually reduce performance relative to the optimized unfused path.

### 4.6 Multi-GPU Implementation

The multi-GPU framework distributes the convolution across all available NVIDIA GPUs using the following steps:

1. **Query devices:** Determine the number of available GPUs with `cudaGetDeviceCount`.
2. **Partition rows:** Divide the image into horizontal strips, one per GPU, with halo overlap on strip boundaries (Section 2.4.2).
3. **Initialize devices:** For each GPU, set the active device with `cudaSetDevice`, create a CUDA stream, allocate device memory, and asynchronously copy the input chunk (including halos) and the filter kernel.
4. **Launch kernels:** Copy the filter to each device's constant memory, then launch the V2 kernel on each GPU.
5. **Collect results:** Asynchronously copy each GPU's output rows (excluding halos) back to the host.
6. **Synchronize and clean up:** Wait for all streams to complete, free device memory, and destroy streams.

The host code iterates over GPUs sequentially to enqueue work, but the actual execution on each GPU proceeds concurrently thanks to CUDA streams.

### 4.7 Benchmarking Methodology

Performance is measured differently for CPU and GPU to account for their distinct execution models:

- **CPU timing:** Uses `std::chrono::high_resolution_clock` to measure wall-clock time. Each benchmark is averaged over 3 iterations.

- **GPU timing:** Uses CUDA events (`cudaEventRecord` / `cudaEventElapsedTime`), which measure time on the GPU's internal clock. This avoids including CPU-side overhead in the measurement. Each benchmark includes 1 warmup iteration (to prime caches and initialize the GPU driver), followed by 10 timed iterations. Results are averaged.

The `CudaTimer` class encapsulates GPU timing:

```cpp
class CudaTimer {
    void start() { cudaEventRecord(start_); }
    void stop()  { cudaEventRecord(stop_);
                   cudaEventSynchronize(stop_); }
    float elapsed_ms() const {
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};
```

All benchmarks use synthetic random noise images generated with a fixed seed (42) for reproducibility. Speedup is computed as `CPU_time / GPU_time`.

The benchmarks sweep four dimensions:

1. **Image size:** 256x256 to 4096x4096 (Gaussian 3x3, 16x16 blocks)
2. **Kernel size:** 3x3, 5x5, 7x7 (1024x1024 image, 16x16 blocks)
3. **Block configuration:** 8x8, 16x16, 32x8, 32x16 (1024x1024 image, Gaussian 5x5)
4. **Fused vs. unfused:** Gaussian + Sobel pipeline (512x512 to 2048x2048)

### 4.8 Correctness Verification

Correctness is established by comparing GPU outputs against the CPU reference output using the maximum absolute error metric:

```cpp
float max_abs_error(const vector<float>& a, const vector<float>& b) {
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        max_err = max(max_err, abs(a[i] - b[i]));
    return max_err;
}
```

A 64x64 checkerboard test image is convolved with each filter on both CPU and GPU. The test passes if the maximum absolute error is below 1e-5 for V1/V2 and below 1e-4 for V3 (the fused kernel accumulates more floating-point operations, allowing slightly higher numerical tolerance). The multi-GPU output is compared against the single-GPU V2 output using the same threshold.

Seven filter configurations are tested: Gaussian 3x3/5x5/7x7, Sobel X 3x3, Laplacian 3x3, and Box Blur 3x3/5x5.

---

## 5. Results

### 5.1 Hardware and Software Environment

| Property | Value |
|----------|-------|
| GPU | [TODO: GPU model] |
| GPU Architecture | [TODO: e.g., Pascal (sm_61)] |
| CUDA Cores | [TODO] |
| GPU Memory | [TODO] |
| Memory Bandwidth | [TODO] |
| Shared Memory per SM | [TODO] |
| CUDA Toolkit Version | [TODO] |
| Host CPU | [TODO] |
| Host RAM | [TODO] |
| Operating System | [TODO] |
| Compiler (Host) | g++ with C++20 |
| Compiler (Device) | nvcc with C++17 |

### 5.2 Correctness Results

| Filter | Size | V1 Max Error | V2 Max Error | V3 Max Error | Status |
|--------|------|-------------|-------------|-------------|--------|
| Gaussian | 3x3 | [TODO] | [TODO] | [TODO] | [TODO] |
| Gaussian | 5x5 | [TODO] | [TODO] | N/A | [TODO] |
| Gaussian | 7x7 | [TODO] | [TODO] | N/A | [TODO] |
| Sobel X | 3x3 | [TODO] | [TODO] | N/A | [TODO] |
| Laplacian | 3x3 | [TODO] | [TODO] | N/A | [TODO] |
| Box Blur | 3x3 | [TODO] | [TODO] | N/A | [TODO] |
| Box Blur | 5x5 | [TODO] | [TODO] | N/A | [TODO] |

**Multi-GPU Correctness:**

| Image Size | Kernel | GPUs Used | Max Error vs Single-GPU | Status |
|-----------|--------|-----------|------------------------|--------|
| 1024x1024 | Gaussian 3x3 | [TODO] | [TODO] | [TODO] |

### 5.3 Image Size Sweep (Gaussian 3x3, 16x16 Blocks)

| Image Size | CPU (ms) | V1 (ms) | V2 (ms) | Speedup V1 | Speedup V2 |
|------------|----------|---------|---------|------------|------------|
| 256x256 | [TODO] | [TODO] | [TODO] | [TODO]x | [TODO]x |
| 512x512 | [TODO] | [TODO] | [TODO] | [TODO]x | [TODO]x |
| 1024x1024 | [TODO] | [TODO] | [TODO] | [TODO]x | [TODO]x |
| 2048x2048 | [TODO] | [TODO] | [TODO] | [TODO]x | [TODO]x |
| 4096x4096 | [TODO] | [TODO] | [TODO] | [TODO]x | [TODO]x |

<!-- Suggested figure: Log-log plot of execution time vs. image size with lines for CPU, V1, V2 -->

### 5.4 Kernel Size Sweep (1024x1024 Image, 16x16 Blocks)

| Kernel Size | CPU (ms) | V1 (ms) | V2 (ms) | V2/V1 Improvement |
|-------------|----------|---------|---------|-------------------|
| 3x3 | [TODO] | [TODO] | [TODO] | [TODO]x |
| 5x5 | [TODO] | [TODO] | [TODO] | [TODO]x |
| 7x7 | [TODO] | [TODO] | [TODO] | [TODO]x |

<!-- Suggested figure: Grouped bar chart comparing V1 and V2 execution times for each kernel size -->

### 5.5 Block Size Sweep (1024x1024, Gaussian 5x5)

| Block Config | Threads/Block | V1 (ms) | V2 (ms) | V2/V1 Improvement |
|-------------|---------------|---------|---------|-------------------|
| 8x8 | 64 | [TODO] | [TODO] | [TODO]x |
| 16x16 | 256 | [TODO] | [TODO] | [TODO]x |
| 32x8 | 256 | [TODO] | [TODO] | [TODO]x |
| 32x16 | 512 | [TODO] | [TODO] | [TODO]x |

<!-- Suggested figure: Bar chart of V2 execution time for each block configuration, highlighting the optimal -->

### 5.6 Fused vs. Unfused Comparison (Gaussian + Sobel, 16x16 Blocks)

| Image Size | Unfused V2 Pipeline (ms) | Fused V3 (ms) | Speedup (Unfused/Fused) |
|------------|--------------------------|---------------|-------------------------|
| 512x512 | [TODO] | [TODO] | [TODO]x |
| 1024x1024 | [TODO] | [TODO] | [TODO]x |
| 2048x2048 | [TODO] | [TODO] | [TODO]x |

<!-- Suggested figure: Paired bar chart (unfused vs. fused) for each image size -->

### 5.7 Multi-GPU Scaling

| Configuration | Time (ms) | Speedup vs Single GPU |
|--------------|-----------|----------------------|
| 1 GPU | [TODO] | 1.00x |
| [TODO] GPUs | [TODO] | [TODO]x |

### 5.8 Discussion of Results

[TODO: Discuss the following points based on actual benchmark data]

**Image size scaling:** How does speedup change as the image grows? Larger images should show higher speedups as GPU occupancy improves and the fixed overhead of kernel launch and data transfer is amortized over more computation.

**Kernel size impact:** How does the V2/V1 improvement ratio change with kernel size? The shared memory data reuse factor increases with larger kernels, as each tile element is read more times during convolution.

**Block configuration:** Why does the 32x8 configuration outperform others? With 32 threads along the x-axis, each row of a thread block aligns perfectly with a warp, ensuring coalesced global memory access during the tile loading phase.

**Fusion trade-offs:** Why is the fused V3 kernel slower than the unfused V2 pipeline? The current V3 implementation lacks shared memory optimization, so the reduction in kernel launches is outweighed by excessive global memory traffic (81 reads per pixel vs. tiled access in V2).

**Achieved bandwidth:** [TODO: Calculate effective memory bandwidth for V2 and compare against the GPU's theoretical peak. Formula: (bytes_read + bytes_written) / time.]

---

## 6. Conclusion

### 6.1 Summary of Findings

[TODO: Summarize key results once benchmarks are complete. Expected points:]

- The shared memory tiled kernel (V2) achieves significant speedup over the CPU baseline and meaningful improvement over the naive GPU kernel (V1).
- Speedup scales with image size, as larger images better utilize the GPU's massive parallelism.
- V2's advantage over V1 grows with kernel size, confirming that shared memory data reuse becomes more valuable as the computation-to-load ratio increases.
- The 32x8 block configuration optimizes for warp-aligned memory coalescing.
- Kernel fusion (V3) demonstrates that reducing kernel launches is not sufficient without also optimizing memory access patterns.

### 6.2 Lessons Learned

- **Memory hierarchy matters more than raw parallelism.** The V1 kernel already uses thousands of threads, but its performance is limited by global memory bandwidth. V2's shared memory tiling provides substantial improvement by exploiting data locality, without launching more threads.

- **Shared memory tiling is the most impactful optimization for stencil computations.** The data reuse factor of tiling grows with kernel size, making it increasingly effective for larger filters.

- **Block shape matters as much as block size.** A 32x8 block and a 16x16 block have the same number of threads (256), but the 32-wide block aligns with warp boundaries for coalesced memory access, resulting in measurably better performance.

- **Kernel fusion has prerequisites.** Fusing operations into a single kernel reduces launch overhead and eliminates intermediate memory writes, but only if the fused kernel's memory access pattern is also optimized. An unoptimized fused kernel can be slower than an optimized multi-kernel pipeline.

- **Template metaprogramming enables zero-cost abstraction on GPUs.** By making kernel size and block dimensions template parameters, the compiler can fully unroll loops and statically size shared memory allocations, producing optimal code for each configuration.

### 6.3 Future Work

- **Optimize the fused kernel (V3):** Add shared memory tiling to the fused Gaussian+Sobel kernel to realize the benefits of both fusion and data reuse.
- **Half-precision arithmetic:** Explore FP16 computation on architectures that support it (Volta and newer) to potentially double throughput for applications that tolerate reduced precision.
- **Separable filter decomposition:** Gaussian filters are separable, meaning a 2D K x K convolution can be decomposed into two 1D passes (one horizontal, one vertical) with O(2K) operations instead of O(K^2). This could significantly accelerate 5x5 and 7x7 Gaussian filters.
- **Profiling with NVIDIA Nsight Compute:** Use hardware performance counters to measure achieved memory bandwidth, occupancy, warp efficiency, and shared memory bank conflicts, providing deeper insight into kernel bottlenecks.
- **Benchmark on newer architectures:** Test on Ampere (RTX 3000), Ada (RTX 4000), or Hopper GPUs to observe the impact of larger shared memory, higher bandwidth, and architectural improvements.
- **Multi-node distribution:** Extend the multi-GPU framework with MPI for distribution across multiple machines, adding a network communication layer to the current shared-memory multi-GPU approach.

---

## 7. References

1. NVIDIA Corporation. *CUDA C++ Programming Guide*. NVIDIA Developer Documentation. https://docs.nvidia.com/cuda/cuda-c-programming-guide/

2. Kirk, D. B. and Hwu, W. W. *Programming Massively Parallel Processors: A Hands-on Approach*. 4th Edition. Morgan Kaufmann, 2022.

3. NVIDIA Corporation. *GeForce GTX 1050 Ti Specifications*. https://www.nvidia.com/en-us/geforce/graphics-cards/10-series/geforce-gtx-1050-ti/

4. Gonzalez, R. C. and Woods, R. E. *Digital Image Processing*. 4th Edition. Pearson, 2018.

5. NVIDIA Corporation. *Nsight Compute Documentation*. https://docs.nvidia.com/nsight-compute/
