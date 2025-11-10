// nvcc -O3 -std=c++17 -arch=sm_80 graph_wave_params.cu -o graph_wave_params
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    std::exit(1); \
  } \
} while (0)

__global__ void rhs_kernel(int n, double alpha, const double* __restrict__ u, double* __restrict__ rhs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) rhs[i] = alpha * u[i]; // toy RHS (e.g., stand-in for M^{-1}(-K u + f))
}

__global__ void wave_step_kernel(int n, double dt2,
                                 const double* __restrict__ u_prev,
                                 const double* __restrict__ u_cur,
                                 const double* __restrict__ rhs,
                                 double* __restrict__ u_next) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // u_{n+1} = 2*u_n - u_{n-1} + dt^2 * rhs(u_n)
    u_next[i] = 2.0 * u_cur[i] - u_prev[i] + dt2 * rhs[i];
  }
}

int main(int argc, char** argv) {
  const int    n      = (argc > 1) ? std::atoi(argv[1]) : 1<<20;   // 1M default
  const int    steps  = (argc > 2) ? std::atoi(argv[2]) : 100;     // time steps
  const double alpha  = 0.1;
  const double dt2    = 1e-4;

  printf("n=%d, steps=%d\n", n, steps);

  // --- allocate device buffers (three time levels + rhs)
  double *d_u_prev, *d_u_cur, *d_u_next, *d_rhs;
  CUDA_CHECK(cudaMalloc(&d_u_prev, n*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_u_cur,  n*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_u_next, n*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_rhs,    n*sizeof(double)));

  // init some values
  std::vector<double> h_init(n);
  for (int i=0;i<n;++i) h_init[i] = std::sin(0.001*i);
  CUDA_CHECK(cudaMemcpy(d_u_prev, h_init.data(), n*sizeof(double), cudaMemcpyHostToDevice));
  for (int i=0;i<n;++i) h_init[i] = std::sin(0.001*i)*1.1;
  CUDA_CHECK(cudaMemcpy(d_u_cur,  h_init.data(), n*sizeof(double), cudaMemcpyHostToDevice));

  // launch config
  dim3 block(256);
  dim3 grid( (n + block.x - 1) / block.x );

  // --- 1) Build graph programmatically (we keep node handles)
  cudaGraph_t graph;
  CUDA_CHECK(cudaGraphCreate(&graph, 0));

  // Node A: rhs_kernel
  cudaKernelNodeParams rhs_params{};
  void* rhs_args[] = { (void*)&n, (void*)&alpha, (void*)&d_u_cur, (void*)&d_rhs };
  rhs_params.func            = (void*)rhs_kernel;
  rhs_params.gridDim         = grid;
  rhs_params.blockDim        = block;
  rhs_params.sharedMemBytes  = 0;
  rhs_params.kernelParams    = rhs_args;
  rhs_params.extra           = nullptr;

  cudaGraphNode_t rhs_node;
  CUDA_CHECK(cudaGraphAddKernelNode(&rhs_node, graph, nullptr, 0, &rhs_params));

  // Node B: wave_step_kernel depends on A
  cudaKernelNodeParams step_params{};
  void* step_args[] = { (void*)&n, (void*)&dt2,
                        (void*)&d_u_prev, (void*)&d_u_cur, (void*)&d_rhs, (void*)&d_u_next };
  step_params.func           = (void*)wave_step_kernel;
  step_params.gridDim        = grid;
  step_params.blockDim       = block;
  step_params.sharedMemBytes = 0;
  step_params.kernelParams   = step_args;
  step_params.extra          = nullptr;

  cudaGraphNode_t step_node;
  CUDA_CHECK(cudaGraphAddKernelNode(&step_node, graph, &rhs_node, 1, &step_params));

  // --- 2) Instantiate executable graph
  cudaGraphExec_t graphExec;
  CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Timing (optional)
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // --- 3) Time stepping loop:
  // swap (u_prev, u_cur, u_next) each step and UPDATE PARAMS for nodes
  double *u_prev = d_u_prev, *u_cur = d_u_cur, *u_next = d_u_next;

  for (int k=0; k<steps; ++k) {
    // Update node A (rhs) to use current u_cur -> rhs
    void* rhs_args_k[] = { (void*)&n, (void*)&alpha, (void*)&u_cur, (void*)&d_rhs };
    cudaKernelNodeParams rhs_updated = rhs_params;
    rhs_updated.kernelParams = rhs_args_k;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(graphExec, rhs_node, &rhs_updated));

    // Update node B (wave step) with (u_prev, u_cur, rhs, u_next)
    void* step_args_k[] = { (void*)&n, (void*)&dt2,
                            (void*)&u_prev, (void*)&u_cur, (void*)&d_rhs, (void*)&u_next };
    cudaKernelNodeParams step_updated = step_params;
    step_updated.kernelParams = step_args_k;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(graphExec, step_node, &step_updated));

    // Launch the graph for this time step
    CUDA_CHECK(cudaGraphLaunch(graphExec, stream));

    // Optionally sync less often; here we allow overlap within stream
    // Rotate buffers: (u_prev,u_cur,u_next) <- (u_cur,u_next,u_prev)
    double* tmp = u_prev;
    u_prev = u_cur;
    u_cur  = u_next;
    u_next = tmp;
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms=0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  printf("Total time: %.3f ms  (avg per step: %.3f ms)\n", ms, ms/steps);

  // Cleanup
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaGraphExecDestroy(graphExec));
  CUDA_CHECK(cudaFree(d_rhs));
  CUDA_CHECK(cudaFree(d_u_next));
  CUDA_CHECK(cudaFree(d_u_cur));
  CUDA_CHECK(cudaFree(d_u_prev));
  return 0;
}
