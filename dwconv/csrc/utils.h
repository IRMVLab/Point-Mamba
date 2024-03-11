// --------------------------------------------------------
// Octree-based Sparse Convolutional Neural Networks
// Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
// Licensed under The MIT License [see LICENSE for details]
// Written by Peng-Shuai Wang
// --------------------------------------------------------

#include <cuda.h>
#include <cuda_runtime.h>

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition)                                        \
  do {                                                               \
    cudaError_t error = condition;                                   \
    CHECK(error == cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                       \
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// CUDA: number of threads per block
constexpr int kCudaThreadsNum = 512;

// CUDA: number of blocks for threads.
inline int CudaGetBlocks(const int N) {
  return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}
