// include/queues/taskmanager_gpu.cuh
#pragma once

#include <cuda_runtime.h>
#include "job_definitions.cuh"
#include "broker.cuh"  // your queue implementation

namespace ASC_HPC {

  // adapt template parameters if broker.cuh uses different ones:
  using GPUQueue = BrokerQueue<1024, Task, 10000>;

  extern __device__ GPUQueue d_queue;
  extern __device__ int      d_doneCounter;
  extern __device__ int      d_stopFlag;

  // Host API
  void StartWorkersGPU(int blocks, int threadsPerBlock);
  void StopWorkersGPU();

  // Enqueue n tasks for a given job (device-side job pointer!)
  void EnqueueJobTasksGPU(Job* d_job, int n);

  // Wait until at least expectedDone tasks have run
  void WaitForAllGPU(int expectedDone);

} // namespace ASC_HPC
