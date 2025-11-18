// include/queues/taskmanager_gpu.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "queues/taskmanager_gpu.cuh"

namespace ASC_HPC {

  // --- small helper for error checking ---
  inline void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
      printf("CUDA ERROR in %s: %s (%d)\n",
             what, cudaGetErrorString(err), (int)err);
      fflush(stdout);
      // hard abort so we see it clearly
      std::abort();
    }
  }

  // Single global device counter
  __device__ int d_doneCounter = 0;

  // Very simple test kernel: runs on GPU, bumps the counter 16 times
  __global__ void test_kernel(int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      for (int i = 0; i < n; ++i) {
        printf("test_kernel: i=%d\n", i);
        atomicAdd(&d_doneCounter, 1);
      }
    }
  }

  // ------- Host API: no queues, no jobs, just this counter test -------

  void StartWorkersGPU(int /*blocks*/, int /*threadsPerBlock*/) {
    // reset counter on device
    int zero = 0;
    checkCuda(cudaMemcpyToSymbol(d_doneCounter, &zero, sizeof(int)),
              "cudaMemcpyToSymbol(d_doneCounter)");

    // launch test kernel once
    test_kernel<<<1, 1>>>(16);

    // check launch error immediately
    checkCuda(cudaGetLastError(), "kernel launch (test_kernel)");

    // (don't sync here; WaitForAllGPU will hit it via memcpy)
  }

  void WaitForAllGPU(int expectedDone) {
    int done  = 0;
    int iter  = 0;
    int const maxIters = 200000;

    while (done < expectedDone && iter < maxIters) {
      checkCuda(cudaMemcpyFromSymbol(&done, d_doneCounter, sizeof(int)),
                "cudaMemcpyFromSymbol(d_doneCounter)");

      if (iter % 1000 == 0) {
        printf("WaitForAllGPU: iter=%d done=%d / %d\n",
               iter, done, expectedDone);
        fflush(stdout);
      }
      ++iter;
    }

    if (done < expectedDone) {
      printf("WaitForAllGPU: TIMEOUT, done=%d / %d\n", done, expectedDone);
    } else {
      printf("WaitForAllGPU: completed, done=%d\n", done);
    }
  }

  void StopWorkersGPU() {
    // just make sure any kernel is finished
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");
  }

} // namespace ASC_HPC
