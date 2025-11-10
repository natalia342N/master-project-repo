#include <cstdio>
#include <cuda_runtime.h>
#include "timer.hpp"   

__global__ void init_fill(float *x, float val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = val;
}

__global__ void k1(float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = a[i] + 0.5f * b[i];
}
__global__ void k2(float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = 1.0001f * a[i];
}
__global__ void k3(float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += 1.0f;
}

int main() {
  const int N = 1'000'000;       // 1 million
  const int R = 1;               // one repeat
  dim3 block(256), grid((N + block.x - 1) / block.x);

  cudaStream_t stream; cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  float *da=nullptr, *db=nullptr;
  cudaMalloc(&da, N*sizeof(float));
  cudaMalloc(&db, N*sizeof(float));

  cudaFree(0);
  init_fill<<<grid, block, 0, stream>>>(da, 1.0f, N);
  init_fill<<<grid, block, 0, stream>>>(db, 2.0f, N);
  cudaStreamSynchronize(stream);

  // NO-GRAPH: launch 3 kernels once 
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  Timer wallA; wallA.reset();
  cudaEventRecord(e0, stream);
  k1<<<grid, block, 0, stream>>>(da, db, N);
  k2<<<grid, block, 0, stream>>>(da, N);
  k3<<<grid, block, 0, stream>>>(da, N);
  cudaEventRecord(e1, stream);
  cudaEventSynchronize(e1);
  float gpu_ms_A = 0.0f; cudaEventElapsedTime(&gpu_ms_A, e0, e1);
  long long wall_us_A = (long long)(1e6 * wallA.get());

  // GRAPH: capture those same 3 kernels, then one launch 
  cudaGraph_t g=nullptr; cudaGraphExec_t ge=nullptr;

  Timer t_cap; t_cap.reset();
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  k1<<<grid, block, 0, stream>>>(da, db, N);
  k2<<<grid, block, 0, stream>>>(da, N);
  k3<<<grid, block, 0, stream>>>(da, N);
  cudaStreamEndCapture(stream, &g);
  long long cap_us = (long long)(1e6 * t_cap.get());

  Timer t_inst; t_inst.reset();
  cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
  long long inst_us = (long long)(1e6 * t_inst.get());

  Timer t_up; t_up.reset();
  cudaGraphUpload(ge, stream);
  long long up_us = (long long)(1e6 * t_up.get());

  // launch once
  cudaEventRecord(e0, stream);
  Timer wallB; wallB.reset();
  cudaGraphLaunch(ge, stream);
  cudaEventRecord(e1, stream);
  cudaEventSynchronize(e1);
  float gpu_ms_B = 0.0f; cudaEventElapsedTime(&gpu_ms_B, e0, e1);
  long long wall_us_B = (long long)(1e6 * wallB.get());

  long long one_use_total_us = cap_us + inst_us + up_us + wall_us_B;

  printf("N = %d elements, R = %d (one capture, one launch)\n", N, R);
  printf("\nA) no-graph (3 separate launches):\n");
  printf("   GPU time (events) : %.3f ms\n", gpu_ms_A);
  printf("   CPU wall (Timer)  : %lld us\n", wall_us_A);

  printf("\nB) graph build & run once:\n");
  printf("   capture            : %lld us\n", cap_us);
  printf("   instantiate        : %lld us\n", inst_us);
  printf("   upload             : %lld us\n", up_us);
  printf("   graph launch GPU   : %.3f ms\n", gpu_ms_B);
  printf("   graph launch wall  : %lld us\n", wall_us_B);
  printf("   one-use total      : %lld us  (cap+inst+upload+launch)\n", one_use_total_us);

  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
  cudaFree(da); cudaFree(db); cudaStreamDestroy(stream);
  return 0;
}
