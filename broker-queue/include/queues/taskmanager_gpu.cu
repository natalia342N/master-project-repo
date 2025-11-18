// include/queues/taskmanager_gpu.cu
#include "taskmanager_gpu.cuh"
#include "job_kernels.cuh"   // brings in Task::execute + job funcs

namespace ASC_HPC {

  // ---- Device globals ----
  __device__ GPUQueue d_queue;
  __device__ int      d_doneCounter = 0;
  __device__ int      d_stopFlag    = 0;

  // ============================================================
  //  Worker kernel: dequeue Task -> Task::execute()
  // ============================================================

  __global__ void worker_kernel() {
    // init queue + counters once
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      d_queue.init();       // your BrokerQueue::init()
      d_doneCounter = 0;
      d_stopFlag    = 0;
      current_phase = 0;    // start with job 0
    }
    __syncthreads();

    Task t;

    while (true) {
      if (atomicAdd(&d_stopFlag, 0) != 0)
        break;

      bool got = false;

      // one thread per block dequeues a Task from BrokerQueue
      if (threadIdx.x == 0) {
        got = d_queue.dequeue(t);   // your BrokerQueue::dequeue(Task&)
      }

      got = __syncthreads_or(got);
      if (!got) {
        // no tasks at the moment
        continue;
      }

      __syncthreads();  // make sure all threads see 't'

      // cooperative execution: each thread calls t.execute(),
      // but the internal job->func() will typically use threadIdx/grid-stride
      t.execute();

      __syncthreads();
      if (threadIdx.x == 0) {
        atomicAdd(&d_doneCounter, 1);
      }
    }
  }

  // ============================================================
  //  Helper kernel: enqueue Task objects on device
  // ============================================================

  __global__ void enqueue_tasks_kernel(Job* job, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Task t;
    t.job = job;
    t.i   = idx;
    d_queue.enqueue(t);   // your BrokerQueue::enqueue(Task&)
  }

  // ============================================================
  //  Host-side symbol access
  // ============================================================

  static int* h_doneCounter_ptr = nullptr;
  static int* h_stopFlag_ptr    = nullptr;

  static void init_symbol_pointers() {
    static bool inited = false;
    if (inited) return;

    cudaGetSymbolAddress((void**)&h_doneCounter_ptr, d_doneCounter);
    cudaGetSymbolAddress((void**)&h_stopFlag_ptr,    d_stopFlag);
    inited = true;
  }

  // ============================================================
  //  Host API implementation
  // ============================================================

  void StartWorkersGPU(int blocks, int threadsPerBlock) {
    init_symbol_pointers();
    worker_kernel<<<blocks, threadsPerBlock>>>();
    // no sync: persistent kernel stays alive
  }

  void StopWorkersGPU() {
    init_symbol_pointers();
    int one = 1;
    cudaMemcpy(h_stopFlag_ptr, &one, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();  // wait for worker_kernel to finish
  }

  void EnqueueJobTasksGPU(Job* d_job, int n) {
    int blockSize = 128;
    int gridSize  = (n + blockSize - 1) / blockSize;
    enqueue_tasks_kernel<<<gridSize, blockSize>>>(d_job, n);
    cudaDeviceSynchronize();  // ensure they've been enqueued
  }

  void WaitForAllGPU(int expectedDone) {
    init_symbol_pointers();
    int done = 0;
    while (done < expectedDone) {
      cudaMemcpy(&done, h_doneCounter_ptr, sizeof(int),
                 cudaMemcpyDeviceToHost);
      // optional: sleep/yield here
    }
  }

} // namespace ASC_HPC
