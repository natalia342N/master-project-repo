// source/main_taskmanager.cu

#include "queues/taskmanager_gpu.cuh"
#include "queues/job_definitions.cuh"
#include "queues/job_kernels.cuh"

using namespace ASC_HPC;

int main() {
  // Allocate data and counters on device
  float* d_data = nullptr;
  cudaMalloc(&d_data, 3 * 1024 * sizeof(float));  // for daxpy + multiply etc.

  int* d_completed0 = nullptr;
  int* d_completed1 = nullptr;
  cudaMalloc(&d_completed0, sizeof(int));
  cudaMalloc(&d_completed1, sizeof(int));
  cudaMemset(d_completed0, 0, sizeof(int));
  cudaMemset(d_completed1, 0, sizeof(int));

  // Get device addresses of global jobs
  Job* d_job0_ptr = nullptr;
  Job* d_job1_ptr = nullptr;
  cudaGetSymbolAddress((void**)&d_job0_ptr, device_job);
  cudaGetSymbolAddress((void**)&d_job1_ptr, device_job2);

  // Setup job 0 (daxpy) and job 1 (multiply) on device
  Job h_job0, h_job1;
  h_job0.func      = daxpy_job_func;
  h_job0.data      = (void*)d_data;
  h_job0.n         = 8;
  h_job0.completed = d_completed0;

  h_job1.func      = multiply_job_func;
  h_job1.data      = (void*)d_data;
  h_job1.n         = 8;
  h_job1.completed = d_completed1;

  cudaMemcpy(d_job0_ptr, &h_job0, sizeof(Job), cudaMemcpyHostToDevice);
  cudaMemcpy(d_job1_ptr, &h_job1, sizeof(Job), cudaMemcpyHostToDevice);

  // Start workers
  StartWorkersGPU(/*blocks=*/80, /*threadsPerBlock=*/128);

  // Enqueue tasks for both jobs
  EnqueueJobTasksGPU(d_job0_ptr, h_job0.n);
  EnqueueJobTasksGPU(d_job1_ptr, h_job1.n);

  // Wait for all tasks (job0.n + job1.n)
  WaitForAllGPU(h_job0.n + h_job1.n);

  // Stop workers
  StopWorkersGPU();

  // TODO: copy back results, check correctness …

  cudaFree(d_data);
  cudaFree(d_completed0);
  cudaFree(d_completed1);
  return 0;
}
