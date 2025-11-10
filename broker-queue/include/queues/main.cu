#include <cstdio>
#include <cuda_runtime.h>
#include "taskmanager_gpu.cu"
#include "job_kernels.cuh"

__device__ void* job_data_ptr;

__device__ int* device_completed_ptr;
__device__ int* device_completed_ptr2;


#include "job_structs.cuh"

__global__ void init_job_and_enqueue() {
    printf("Inside enqueue kernel\n");

    float* data = reinterpret_cast<float*>(job_data_ptr);
    for (int i = 0; i < 1024; ++i) {
        data[i] = 1.0f;
        data[1024 + i] = 2.0f;
    }

    device_job.func = daxpy_job_func;
    device_job.data = job_data_ptr;
    device_job.n = 8;
    device_job.completed = device_completed_ptr;

    for (int i = 0; i < device_job.n; ++i) {
        Task t;
        t.job = &device_job;
        t.i = i;
        bool ok = globalQueue.enqueue(t);
        printf("Task %d enqueue: %s\n", i, ok ? "OK" : "FAIL");
    }
}

__global__ void enqueue_second_job() {
    printf("Inside enqueue_second_job kernel\n");

    device_job2.func = multiply_job_func;
    device_job2.data = job_data_ptr;
    device_job2.n = 8;
    device_job2.completed = device_completed_ptr2;

    for (int i = 0; i < device_job2.n; ++i) {
        Task t;
        t.job = &device_job2;
        t.i = i;
        bool ok = globalQueue.enqueue(t);
        printf("Job2 Task %d enqueue: %s\n", i, ok ? "OK" : "FAIL");
    }
}

int main() {
    int* completed1;
    int* completed2;
    
    cudaMalloc(&completed1, sizeof(int));
    cudaMalloc(&completed2, sizeof(int));
    cudaMemset(completed1, 0, sizeof(int));
    cudaMemset(completed2, 0, sizeof(int));
    
    cudaMemcpyToSymbol(device_completed_ptr, &completed1, sizeof(int*));     // Job 1
    cudaMemcpyToSymbol(device_completed_ptr2, &completed2, sizeof(int*));    // Job 2

    void* raw_data;
    cudaMalloc(&raw_data, sizeof(float) * 3072);
    cudaMemcpyToSymbol(job_data_ptr, &raw_data, sizeof(void*));

    // 1. Initialize the queue
    init_queue<<<1,1>>>();
    cudaDeviceSynchronize();
    
    // 2. Enqueue Job 1: DAXPY
    init_job_and_enqueue<<<1,1>>>();
    
    // 3. Enqueue Job 2: Multiply
    enqueue_second_job<<<1,1>>>();
    
    // 4. Process all tasks in one kernel launch
    cudaDeviceSynchronize();
    printf("Launching process_tasks_kernel...\n");

    process_tasks_kernel<<<1, 16>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    printf("Finished kernel.\n");

    cudaDeviceSynchronize();


    cudaFree(raw_data);
    cudaFree(completed1);
    cudaFree(completed2);

    printf("Host: Program ran to end.\n");
    return 0;
}