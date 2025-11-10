#pragma once
#include "job_structs.cuh"
#include "broker.cuh"

__device__ BrokerQueue<1024, Task, 10000> globalQueue;

__global__ void init_queue();
__global__ void process_tasks_kernel();