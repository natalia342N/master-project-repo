// file: cuda_events_min.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void MyKernel(float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = i * 0.001f;
}

int main() {
  int N = 1<<20;
  float *d;
  cudaMalloc(&d, N*sizeof(float));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  MyKernel<<<(N+255)/256, 256>>>(d, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms=0;
  cudaEventElapsedTime(&ms, start, stop);
  std::printf("Kernel time: %.3f ms\n", ms);

  cudaFree(d);
  return 0;
}
