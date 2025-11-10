// job_definitions.cuh
#pragma once

struct Job {
    void (*func)(void*, int, int);
    void* data;
    int n;
    int* completed;
};

struct Task {
    Job* job;
    int i;

    __device__ void execute() const;
};
