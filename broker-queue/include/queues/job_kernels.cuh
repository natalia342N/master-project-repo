#pragma once

#include "queues/job_definitions.cuh"

// -------------------------------------------------------------
// Device job functions (no globals referenced here)
// -------------------------------------------------------------

// Job 0: DAXPY-like operation on 1024-element vectors
__device__ void daxpy_job_func(void* raw_data, int i, int n) {
    float* A = reinterpret_cast<float*>(raw_data);
    float* B = A + 1024;
    float alpha = 2.0f;

    int chunk = 1024 / n;
    int start = i * chunk;
    int end   = (i == n - 1) ? 1024 : start + chunk;

    for (int j = start; j < end; ++j)
        A[j] += alpha * B[j];
}

// Job 1: elementwise multiply x * y -> out, each length 1024
__device__ void multiply_job_func(void* data, int i, int n) {
    float* vec = reinterpret_cast<float*>(data);
    float* x   = vec;
    float* y   = vec + 1024;
    float* out = vec + 2048;

    int chunk = 1024 / n;
    int start = i * chunk;
    int end   = start + chunk;

    for (int j = start; j < end; ++j) {
        out[j] = x[j] * y[j];
    }
}
