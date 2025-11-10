#include <cstdio>
#include <cuda_runtime.h>
#include "queues/broker.cuh"

struct Task {
    int value;

    __host__ __device__
    Task() : value(0) {}

    __host__ __device__
    Task(int v) : value(v) {}
};

static const unsigned int CAPACITY = 1 << 12;  // 4096
static const unsigned int MAX_THREADS = 1024;

typedef BrokerQueue<CAPACITY, Task, MAX_THREADS> MyBrokerQueue;

__global__ void initKernel(MyBrokerQueue* q)
{
    q->init();
}

__global__ void produceKernel(MyBrokerQueue* q, int totalTasks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalTasks)
    {
        Task t;          // uses default Task()
        t.value = idx;   // fill with your data
        q->enqueue(t);   // ignoring return bool for brevity
    }
}

__global__ void consumeKernel(MyBrokerQueue* q)
{
    while (true)
    {
        Task t;
        bool hasData;
        q->dequeue(hasData, t);
        if (!hasData) {
            break; 
        }
        printf("GPU Thread %d got task value=%d\n",
               threadIdx.x + blockIdx.x*blockDim.x, t.value);
    }
}

int main()
{
    MyBrokerQueue* dQueue = nullptr;
    cudaMalloc(&dQueue, sizeof(MyBrokerQueue));

    initKernel<<<1,1>>>(dQueue);
    cudaDeviceSynchronize();
    produceKernel<<<1,32>>>(dQueue, 32);
    cudaDeviceSynchronize();
    consumeKernel<<<1,32>>>(dQueue);
    cudaDeviceSynchronize();

    cudaFree(dQueue);
    printf("Done.\n");
    return 0;
}
