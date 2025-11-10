#include <cuda_runtime_api.h>
#include <math.h>

__host__ __device__ float rand_lcg(unsigned int& rng_state)
{
    rng_state = 1664525U * rng_state + 1013904223U;
    return fminf(1.0f, rng_state * (1.0f / 4294967296.0f));
}