#include <cstdio>
#include <cuda_runtime.h>
#include "timer.hpp"  

__global__ void tiny_kernel(int *counter) {
  if (threadIdx.x == 0) atomicAdd(counter, 1);
}

int main(int argc, char** argv) {
  int N_KERNELS = (argc > 1) ? atoi(argv[1]) : 5;
  dim3 blocks(1), threads(1);

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  int *d_counter = nullptr;
  cudaMalloc(&d_counter, sizeof(int));
  cudaMemsetAsync(d_counter, 0, sizeof(int), stream);

  tiny_kernel<<<blocks, threads, 0, stream>>>(d_counter);
  cudaStreamSynchronize(stream);

  //no graph 
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

  Timer wall_no_graph; wall_no_graph.reset();
  cudaEventRecord(e0, stream);
  for (int i = 0; i < N_KERNELS; ++i)
    tiny_kernel<<<blocks, threads, 0, stream>>>(d_counter);
  cudaEventRecord(e1, stream);
  cudaEventSynchronize(e1);
  float gpu_ms_no_graph = 0.f;
  cudaEventElapsedTime(&gpu_ms_no_graph, e0, e1);
  long long wall_us_no_graph = (long long)(wall_no_graph.get() * 1e6);

  //build graph
  cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;

  Timer t_capture; t_capture.reset();
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  for (int i = 0; i < N_KERNELS; ++i)
    tiny_kernel<<<blocks, threads, 0, stream>>>(d_counter);
  cudaStreamEndCapture(stream, &graph);
  long long us_capture = (long long)(t_capture.get() * 1e6);

  Timer t_instantiate; t_instantiate.reset();
  cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  long long us_instantiate = (long long)(t_instantiate.get() * 1e6);

  Timer t_upload; t_upload.reset();
  cudaGraphUpload(exec, stream);
  long long us_upload = (long long)(t_upload.get() * 1e6);

  //launch graph twice
  Timer wall_graph1; wall_graph1.reset();
  cudaEventRecord(e0, stream);
  cudaGraphLaunch(exec, stream);
  cudaEventRecord(e1, stream);
  cudaEventSynchronize(e1);
  float gpu_ms_graph1 = 0.f;
  cudaEventElapsedTime(&gpu_ms_graph1, e0, e1);
  long long wall_us_graph1 = (long long)(wall_graph1.get() * 1e6);

  Timer wall_graph2; wall_graph2.reset();
  cudaEventRecord(e0, stream);
  cudaGraphLaunch(exec, stream);
  cudaEventRecord(e1, stream);
  cudaEventSynchronize(e1);
  float gpu_ms_graph2 = 0.f;
  cudaEventElapsedTime(&gpu_ms_graph2, e0, e1);
  long long wall_us_graph2 = (long long)(wall_graph2.get() * 1e6);

  int h=0; cudaMemcpy(&h, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

  //results
  printf("kernels per batch : %d\n", N_KERNELS);
  printf("final counter     : %d\n", h);

  printf("\n-- no graph --\n");
  printf("GPU time : %.3f ms\n", gpu_ms_no_graph);
  printf("CPU overhead  : %lld us\n", wall_us_no_graph);

  printf("\n-- graph build --\n");
  printf("capture begin-end          : %lld us\n", us_capture);
  printf("instantiate into exe graph      : %lld us\n", us_instantiate);
  printf("upload to gpu        : %lld us\n", us_upload);
  long long one_use_total = us_capture + us_instantiate + us_upload + wall_us_graph1;
  printf("one-use total     : %lld us  (build+first launch)\n", one_use_total);

  printf("\n-- graph launch --\n");
  printf("run #1 GPU        : %.3f ms | wall: %lld us\n", gpu_ms_graph1, wall_us_graph1);
  printf("run #2 GPU        : %.3f ms | wall: %lld us\n", gpu_ms_graph2, wall_us_graph2);

  cudaEventDestroy(e0); cudaEventDestroy(e1);
  cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
  cudaFree(d_counter); cudaStreamDestroy(stream);
  return 0;
}
