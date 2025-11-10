#include "helper.cuh"
#include <type_traits>

template <typename Q>
__global__ void sizeOnDevice(unsigned int* deviceSize)
{
    *deviceSize = sizeof(Q);
}

template <typename Q>
__global__ void placeQueue(void* deviceQueueP)
{
    new (deviceQueueP) Q();
}

template <typename Q>
__global__ void initTest(Q* queueP)
{
    Q& queue = *queueP;
    queue.init();
}

template <typename Q, typename T>
__global__ void launchTest(Q* queueP, unsigned int numIterations, float enqProb, float deqProb, bool waitForData, int* processedCounters)
{
    Q& queue = *queueP;
    int lid = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int processed = 0;

    T a(lid), b;

    float rand_val;
    unsigned int rng_state = lid;

    for (int i = 0; i < numIterations; i++)
    {
        rand_val = rand_lcg(rng_state);
        if (rand_val <= enqProb && tid == 0)
        {
            // Only one thread enqueues (e.g. thread 0 of each block)
            queue.enqueue(a);
        }

        rand_val = rand_lcg(rng_state);
        if (rand_val <= deqProb)
        {
            bool hasData = false;
            if (tid == 0)
            {
                queue.dequeue(hasData, b);
                while (waitForData && !hasData)
                {
                    queue.dequeue(hasData, b);
                }
            }

            // Broadcast success to entire block
            __shared__ bool sharedHasData;
            if (tid == 0) sharedHasData = hasData;
            __syncthreads();

            if (sharedHasData)
            {
                if constexpr (std::is_same<T, DaxpyTask>::value)
                {
                    __shared__ float x[DaxpyTask::SIZE];
                    __shared__ float y[DaxpyTask::SIZE];
                    __shared__ float a;

                    if (tid == 0)
                    {
                        a = b.a;
                        for (int i = 0; i < DaxpyTask::SIZE; ++i)
                        {
                            x[i] = b.x[i];
                            y[i] = b.y[i];
                        }
                    }
                    __syncthreads();

                    for (int i = tid; i < DaxpyTask::SIZE; i += blockSize)
                    {
                        y[i] = a * x[i] + y[i];
                    }

                    __syncthreads();  
                }
                else if constexpr (std::is_same<T, LargerStruct>::value)
                {
                    if (tid == 0)
                    {
                        int result = b.dot();
                        (void)result; // suppress unused warning
                    }
                }

                if (tid == 0)
                    processed++;  // Count only if task actually processed
            }
        }
    }

    if (tid == 0)
        processedCounters[lid] = processed;
}
