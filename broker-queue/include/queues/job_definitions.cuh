// include/queues/job_definitions.cuh
#pragma once

// A "job": some device function + payload + logical parallelism n
struct Job {
    void (*func)(void*, int, int);  // __device__ function pointer
    void* data;                     // device pointer to job-specific data
    int   n;                        // number of logical tasks
    int*  completed;                // device pointer to completion counter
};

// A "task": one logical unit of work (index i) of a job
struct Task {
    Job* job;   // device pointer to the Job
    int  i;     // task index in [0, job->n)

    __device__ void execute() const;
};

// Phase control & the two jobs used in your earlier code
extern __device__ int  current_phase;
extern __device__ Job  device_job;
extern __device__ Job  device_job2;
