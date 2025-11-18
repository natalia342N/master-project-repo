// include/queues/job_kernels.cuh
#pragma once

#include "job_definitions.cuh"

// ---- Job functions (your code, just moved here) ----

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

// ---- Phase-controlled Task::execute implementation ----

__device__ int  current_phase;
__device__ Job  device_job;
__device__ Job  device_job2;

__device__ void Task::execute() const {
    int phase  = atomicAdd(&current_phase, 0);
    int job_id = (job == &device_job) ? 0 : 1;

    if (job_id != phase)
        return;

    // optional debug
    // printf("Executing task %d from job %d\n", i, job_id);

    job->func(job->data, i, job->n);

    int done = atomicAdd(job->completed, 1) + 1;
    if (done == job->n) {
        // printf("All %d tasks complete. Job %d done.\n", job->n, job_id);
        if (job_id == 0) {
            // move to phase 1 when job 0 is done
            atomicExch(&current_phase, 1);
        }
    }
}
