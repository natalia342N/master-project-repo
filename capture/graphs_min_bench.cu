// Minimal CUDA Graphs benchmark: capture / instantiate / first launch / repeat launch
// Build: nvcc -std=c++17 graphs_min_bench.cu -o graphs_min_bench
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <chrono>
#include <cstdlib>

#define CUDA_CHECK(err) do { \
  cudaError_t e = (err); \
  if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while(0)

using clock_tpt = std::chrono::time_point<std::chrono::high_resolution_clock>;
static inline clock_tpt now() { return std::chrono::high_resolution_clock::now(); }
template <class T> static inline double us(T a, T b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count() / 1000.0;
}

// Tiny kernel to give nodes some work
__global__ void empty_kernel() {}

static cudaGraph_t create_parallel_chain(int length, int width, std::vector<cudaStream_t>& streams) {
  // Capture a graph with 'width' branches, each of 'length' empty kernels.
  cudaGraph_t graph = nullptr;

  // Use events to align fork & join during capture
  cudaEvent_t gate;
  CUDA_CHECK(cudaEventCreateWithFlags(&gate, cudaEventDisableTiming));

  CUDA_CHECK(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal));

  // Gate: record on stream 0, all other streams wait → fork alignment
  CUDA_CHECK(cudaEventRecord(gate, streams[0]));
  for (int i = 1; i < width; ++i) {
    CUDA_CHECK(cudaStreamWaitEvent(streams[i], gate, 0));
  }

  // Enqueue length nodes per stream
  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < length; ++j) {
      empty_kernel<<<1, 1, 0, streams[i]>>>();
    }
  }

  // Join: record on each nonzero stream, wait on stream 0
  for (int i = 1; i < width; ++i) {
    CUDA_CHECK(cudaEventRecord(gate, streams[i]));
    CUDA_CHECK(cudaStreamWaitEvent(streams[0], gate, 0));
  }

  CUDA_CHECK(cudaStreamEndCapture(streams[0], &graph));
  CUDA_CHECK(cudaEventDestroy(gate));
  return graph;
}

int main(int argc, char** argv) {
  // Args: [length] [width]
  int length = (argc > 1) ? std::atoi(argv[1]) : 20;
  int width  = (argc > 2) ? std::atoi(argv[2]) : 4;

  // Warm-up context
  CUDA_CHECK(cudaFree(0));

  // Create streams
  std::vector<cudaStream_t> streams(width);
  for (int i = 0; i < width; ++i) CUDA_CHECK(cudaStreamCreate(&streams[i]));

  // -------- capture --------
  auto t0 = now();
  cudaGraph_t graph = create_parallel_chain(length, width, streams);
  auto t1 = now();
  double capture_us = us(t0, t1);

  // -------- instantiate --------
  cudaGraphExec_t exec = nullptr;
  t0 = now();
  CUDA_CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
  t1 = now();
  double instantiate_us = us(t0, t1);

  // -------- first launch (total, wall clock with sync) --------
  t0 = now();
  CUDA_CHECK(cudaGraphLaunch(exec, streams[0]));
  CUDA_CHECK(cudaStreamSynchronize(streams[0]));
  t1 = now();
  double first_launch_total_us = us(t0, t1);

  // -------- repeat launch (total) --------
  t0 = now();
  CUDA_CHECK(cudaGraphLaunch(exec, streams[0]));
  CUDA_CHECK(cudaStreamSynchronize(streams[0]));
  t1 = now();
  double repeat_launch_total_us = us(t0, t1);

  // Output
  std::printf("# CUDA Graphs minimal benchmark\n");
  std::printf("length=%d, width=%d\n", length, width);
  std::printf("capture_us=%.3f\n", capture_us);
  std::printf("instantiate_us=%.3f\n", instantiate_us);
  std::printf("first_launch_total_us=%.3f\n", first_launch_total_us);
  std::printf("repeat_launch_total_us=%.3f\n", repeat_launch_total_us);

  // Cleanup
  CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  for (auto& s : streams) CUDA_CHECK(cudaStreamDestroy(s));

  return 0;
}
