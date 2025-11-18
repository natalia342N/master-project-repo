#include <cstdio>
#include "queues/taskmanager_gpu.cuh"

int main() {
  ASC_HPC::StartWorkersGPU(1, 1);
  ASC_HPC::WaitForAllGPU(16);
  ASC_HPC::StopWorkersGPU();

  printf("Host: finished minimal GPU test.\n");
  return 0;
}
