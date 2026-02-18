# CUDA-Optimized Convolution Stencils

GPU-accelerated 2D convolution with multiple kernel versions of increasing optimization and configurable block sizes.

## Prerequisites

- CUDA Toolkit (nvcc)
- g++ with C++20 support
- An NVIDIA GPU

## Build

```bash
# Full build (requires CUDA)
make CUDA_ARCH=61      # GTX 1050 Ti (Pascal)
make CUDA_ARCH=75      # T4 / RTX 2080 (Turing)
make CUDA_ARCH=86      # RTX 3080 (Ampere)
make CUDA_ARCH=89      # RTX 4090 (Ada)

# CPU-only build (no GPU required)
make cpu_only
```

Find your compute capability at: https://developer.nvidia.com/cuda-gpus

## Run

```bash
./convolution_benchmark        # Full CUDA build
./convolution_benchmark_cpu    # CPU-only build
```

This runs all correctness tests and benchmarks automatically.

## What It Does

Implements 2D image convolution (zero-padded) in four ways:

| Version | Description | Where |
|---------|-------------|-------|
| **CPU** | Sequential baseline, arbitrary kernel sizes (3x3, 5x5, 7x7) | `cpu_convolution.cpp` |
| **GPU V1** | Naive — one thread per pixel, kernel weights from global memory | `cuda_convolution.cu` |
| **GPU V1_const** | Same as V1, but kernel weights from `__constant__` memory | `cuda_convolution.cu` |
| **GPU V2** | Tiled — shared memory for input tile + halo, constant memory kernel | `cuda_convolution.cu` |

All GPU kernels accept configurable block dimensions (8x8, 16x16, 32x8, 32x16, 32x32).

## Parallelization Strategy

### What is parallelized

2D convolution computes each output pixel independently: it multiplies a small filter kernel against the pixel's neighborhood and sums the results. Since every output pixel is independent, this maps naturally to GPU threads — one thread per pixel, with the 2D image tiled into 2D thread blocks.

### Why GPU convolution is fast

1. **Massive parallelism.** A 2048x2048 image has ~4M pixels. The GPU launches 4M threads that execute concurrently across all SMs, while the CPU processes pixels sequentially (or with limited SIMD).

2. **Memory coalescing.** Adjacent threads in a warp read adjacent pixels in memory, enabling coalesced 128-byte transactions instead of scattered loads.

3. **Constant memory caching (V1_const).** The filter kernel is small (e.g. 25 floats for 5x5) and read-only. Placing it in `__constant__` memory means all threads in a warp broadcast-read the same weight from a dedicated cache, eliminating redundant global memory traffic.

4. **Shared memory tiling (V2).** Each thread block loads its input tile (plus a halo for border pixels) into fast on-chip shared memory. Neighbor reads during convolution then hit shared memory (~100x lower latency than global memory) instead of going off-chip. The benefit grows with larger kernels because each pixel touches more neighbors.

### Optimization progression

```
V1 (global)  →  V1_const (+ constant mem kernel)  →  V2 (+ shared mem tiling)
    ↓                    ↓                                   ↓
All reads from      Kernel weights cached           Input tile in shared mem,
global DRAM         in constant cache               kernel in constant cache
```

## Supported Filters

Gaussian blur (3x3, 5x5, 7x7), Sobel X/Y, Laplacian, box blur, sharpen, emboss, identity.

Defined in `filters.cpp`.

## Benchmarks

Results are from two GPUs. Run the benchmark on your own GPU to get updated numbers.

### Correctness

All GPU kernels produce bit-identical output to the CPU baseline (max error = 0.00) on both platforms:

| Filter | V1 | V1_const | V2 |
|--------|----|----|-----|
| Gaussian 3x3, 5x5, 7x7 | PASS | PASS | PASS |
| Sobel X 3x3 | PASS | PASS | PASS |
| Laplacian 3x3 | PASS | PASS | PASS |
| Box blur 3x3, 5x5 | PASS | PASS | PASS |

### Image Size Sweep (Gaussian 3x3)

**GTX 1050 Ti (Pascal, 6 SMs, 112 GB/s)**

| Size | CPU (ms) | V1 (ms) | V1c (ms) | V2 (ms) | Speedup V1 | Speedup V2 |
|------|----------|---------|----------|---------|------------|------------|
| 256x256 | 1.00 | 0.017 | 0.020 | 0.030 | 59x | 34x |
| 512x512 | 3.99 | 0.073 | 0.072 | 0.103 | 54x | 39x |
| 1024x1024 | 16.11 | 0.287 | 0.270 | 0.385 | 56x | 42x |
| 2048x2048 | 65.99 | 1.132 | 1.047 | 1.519 | 58x | 43x |
| 4096x4096 | 265.23 | 4.500 | 4.176 | 5.819 | 59x | 46x |

**Tesla T4 (Turing, 40 SMs, 320 GB/s)**

| Size | CPU (ms) | V1 (ms) | V1c (ms) | V2 (ms) | Speedup V1 | Speedup V2 |
|------|----------|---------|----------|---------|------------|------------|
| 256x256 | 0.90 | 0.014 | 0.017 | 0.020 | 64x | 44x |
| 512x512 | 3.58 | 0.033 | 0.033 | 0.044 | 110x | 81x |
| 1024x1024 | 14.66 | 0.102 | 0.100 | 0.123 | 143x | 120x |
| 2048x2048 | 60.01 | 0.407 | 0.383 | 0.470 | 148x | 128x |
| 4096x4096 | 257.72 | 1.853 | 1.707 | 2.158 | 139x | 119x |

V2 speedup over CPU grows with image size because larger images keep the GPU busier (better occupancy). For small kernels like 3x3, V2's shared memory overhead exceeds the benefit — V1_const is faster. V2 shines with larger kernels (see below). The T4 achieves up to 148x speedup vs 59x on the GTX 1050 Ti, thanks to 6.7x more SMs and 2.9x higher bandwidth.

### Kernel Size Sweep (1024x1024 image)

**GTX 1050 Ti**

| Kernel | CPU (ms) | V1 (ms) | V1c (ms) | V2 (ms) | V2/V1 |
|--------|----------|---------|----------|---------|-------|
| 3x3 | 16.29 | 0.274 | 0.260 | 0.356 | 0.77x |
| 5x5 | 35.73 | 0.600 | 0.556 | 0.402 | 1.49x |
| 7x7 | 67.68 | 0.968 | 0.862 | 0.510 | 1.90x |

**Tesla T4**

| Kernel | CPU (ms) | V1 (ms) | V1c (ms) | V2 (ms) | V2/V1 |
|--------|----------|---------|----------|---------|-------|
| 3x3 | 16.97 | 0.122 | 0.116 | 0.147 | 0.83x |
| 5x5 | 34.99 | 0.249 | 0.216 | 0.222 | 1.12x |
| 7x7 | 66.66 | 0.463 | 0.338 | 0.350 | 1.32x |

For 3x3, the shared memory tiling overhead outweighs the benefit (only 9 neighbor reads per pixel). Starting at 5x5, V2 pulls ahead, and at 7x7 it's 1.90x faster than V1 on the GTX 1050 Ti and 1.32x on the T4. V2's relative gain is smaller on the T4 because its improved caches reduce V1's penalty. V1_const consistently outperforms V1 by 5-27%.

### Block Size Sweep (2048x2048, Gaussian 5x5)

**GTX 1050 Ti**

| Block | Threads | V1 (ms) | V1c (ms) | V2 (ms) | V2/V1 |
|-------|---------|---------|----------|---------|-------|
| 8x8 | 64 | 2.555 | 2.292 | 1.558 | 1.64x |
| 16x16 | 256 | 2.407 | 2.215 | 1.588 | 1.52x |
| 32x8 | 256 | 2.305 | 2.047 | 1.418 | 1.63x |
| 32x16 | 512 | 2.265 | 2.063 | 1.870 | 1.21x |
| 32x32 | 1024 | 2.303 | 2.100 | 2.043 | 1.13x |

**Tesla T4**

| Block | Threads | V1 (ms) | V1c (ms) | V2 (ms) | V2/V1 |
|-------|---------|---------|----------|---------|-------|
| 8x8 | 64 | 1.545 | 1.277 | 0.866 | 1.78x |
| 16x16 | 256 | 0.980 | 0.849 | 0.846 | 1.16x |
| 32x8 | 256 | 0.855 | 0.821 | 0.688 | 1.24x |
| 32x16 | 512 | 0.903 | 0.861 | 0.789 | 1.14x |
| 32x32 | 1024 | 0.973 | 0.886 | 0.898 | 1.08x |

Key observations:
- **32x8 is the fastest V2 config on both GPUs** — 32 threads wide means every warp accesses contiguous memory (perfect coalescing), and the small tile height keeps shared memory usage low, allowing more concurrent blocks per SM.
- **8x8 blocks (64 threads):** Only 2 warps per block — not enough to hide memory latency, resulting in the worst V1 performance.
- **32x32 blocks (1024 threads):** Maximum threads per block, but the large shared memory tile ((32+4)x(32+4) = 1296 floats for 5x5) limits concurrent blocks per SM, making V2 barely faster than V1.
- **The T4 is more sensitive to block shape** for V1 (1.8x spread vs 1.13x on GTX 1050 Ti), suggesting its larger SM count amplifies the occupancy penalty of suboptimal block sizes.

## Project Structure

```
src/
├── main.cpp               # Benchmark harness and correctness tests
├── cpu_convolution.cpp/h  # CPU convolution (arbitrary kernel sizes)
├── cuda_convolution.cu    # GPU kernels V1, V1_const, V2
├── cuda_convolution.cuh   # CUDA interfaces, error checking, timer
├── filters.cpp/h          # Filter kernel definitions
└── image_utils.h          # Synthetic image generators (noise, checkerboard, gradients)
```
