#include "concurrentqueue.h"
#include <cstdio>

__global__ void k() {
    // Device code attempt:
    moodycamel::ConcurrentQueue<int> q; 
    q.enqueue(7);                     
}

int main() {
    k<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
