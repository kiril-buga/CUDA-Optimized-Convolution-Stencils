# CUDA-Optimized Convolution Stencils

GPU-accelerated 2D convolution with three kernel versions of increasing optimization, plus a multi-GPU framework.

## Prerequisites

- CUDA Toolkit (nvcc)
- g++ with C++20 support
- An NVIDIA GPU

## Build

```bash
# Full build (requires CUDA)
make CUDA_ARCH=61      # GTX 1050 Ti (Pascal)
make CUDA_ARCH=75      # RTX 2080 (Turing)
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
| **GPU V1** | Naive - one thread per pixel, global memory reads | `cuda_convolution.cu` |
| **GPU V2** | Tiled - shared memory with halo handling, constant memory kernel | `cuda_convolution.cu` |
| **GPU V3** | Fused - Gaussian blur + Sobel magnitude in a single kernel | `cuda_convolution.cu` |
| **Multi-GPU** | Splits image by rows with halo overlap, stitches on CPU | `cuda_convolution.cu` |

## Supported Filters

Gaussian blur (3x3, 5x5, 7x7), Sobel X/Y, Laplacian, box blur, sharpen, emboss, identity.

Defined in `filters.cpp`.

## Results (NVIDIA GeForce GTX 1050 Ti, 6 SMs, 112 GB/s bandwidth)

### Correctness

All GPU kernels produce bit-identical output to the CPU baseline (error = 0.00):

| Filter | V1 | V2 | V3 (fused) |
|--------|----|----|------------|
| Gaussian 3x3, 5x5, 7x7 | PASS | PASS | PASS |
| Sobel X 3x3 | PASS | PASS | - |
| Laplacian 3x3 | PASS | PASS | - |
| Box blur 3x3, 5x5 | PASS | PASS | - |
| Multi-GPU | PASS (0.00 error vs single GPU) | | |

### Image Size Sweep (Gaussian 3x3)

| Size | CPU (ms) | V1 (ms) | V2 (ms) | Speedup V1 | Speedup V2 |
|------|----------|---------|---------|------------|------------|
| 256x256 | 1.58 | 0.017 | 0.019 | 90x | 83x |
| 512x512 | 6.33 | 0.073 | 0.064 | 87x | 100x |
| 1024x1024 | 25.31 | 0.287 | 0.228 | 88x | 111x |
| 2048x2048 | 100.59 | 1.132 | 0.884 | 89x | 114x |
| 4096x4096 | 403.54 | 4.500 | 3.520 | 90x | 115x |

V2 speedup grows with image size because larger images keep the GPU busier (better occupancy).

### Kernel Size Sweep (1024x1024 image)

| Kernel | CPU (ms) | V1 (ms) | V2 (ms) | V2/V1 |
|--------|----------|---------|---------|-------|
| 3x3 | 25.23 | 0.276 | 0.216 | 1.28x |
| 5x5 | 57.43 | 0.605 | 0.300 | 2.02x |
| 7x7 | 104.61 | 0.967 | 0.436 | 2.22x |

Larger kernels benefit more from shared memory tiling because each pixel reads more neighbors, increasing data reuse per tile.

### Block Size Sweep (1024x1024, Gaussian 5x5)

| Block | V1 (ms) | V2 (ms) | V2/V1 |
|-------|---------|---------|-------|
| 8x8 | 0.652 | 0.296 | 2.20x |
| 16x16 | 0.601 | 0.305 | 1.97x |
| 32x8 | 0.606 | 0.235 | **2.57x** |
| 32x16 | 0.621 | 0.257 | 2.42x |

32x8 is optimal because the 32-wide rows match the warp size, giving perfectly coalesced memory access.

### Fused vs Unfused (Gaussian + Sobel)

| Size | Unfused (ms) | Fused V3 (ms) | Speedup |
|------|-------------|---------------|---------|
| 512x512 | 0.170 | 0.231 | 0.74x |
| 1024x1024 | 0.641 | 0.906 | 0.71x |
| 2048x2048 | 2.510 | 3.596 | 0.70x |

The fused kernel eliminates intermediate global memory writes (1 kernel launch vs 3), but the current implementation uses a simple per-thread approach without shared memory. The unfused path uses the optimized V2 kernel, which is why it's faster here.

## Project Structure

```
src/
├── main.cpp               # Benchmark harness and correctness tests
├── cpu_convolution.cpp/h  # CPU convolution (arbitrary kernel sizes)
├── cuda_convolution.cu    # GPU kernels V1, V2, V3 + multi-GPU
├── cuda_convolution.cuh   # CUDA interfaces, error checking, timer
├── filters.cpp/h          # Filter kernel definitions
└── image_utils.h          # Synthetic image generators (noise, checkerboard, gradients)
```
