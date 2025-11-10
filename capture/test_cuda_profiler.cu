#include <iostream>
#include <ngstd.hpp>
#include <cuda_ngstd.hpp>
#include "cuda_profiler.hpp"

using namespace std;
using namespace ngs_cuda;

__global__ void MyKernel(float *data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        data[i] = sinf(i * 0.001f);
}

int main()
{
    cout << "=== NGSolve CUDA profiler test ===" << endl;

    int N = 1 << 20;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // Label used for NGSolve's host/device timers
    static ngcore::Timer<> t_myregion("MyCudaRegion");

    {
        RegionTimer rt(t_myregion);      // host-side timer
        CudaRegionTimer crt(t_myregion); // device-side timer region

        MyKernel<<<(N + 255) / 256, 256>>>(d_data, N);
        cudaDeviceSynchronize();

        // Copy profiling data from device and reset buffers
        CudaRegionTimer::ProcessTracingData();
    }

    cudaFree(d_data);
    cout << "=== Done ===" << endl;
}
