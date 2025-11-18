#pragma once
#include <cuda_runtime.h>



namespace ASC_HPC {
  void StartWorkersGPU(int blocks, int threadsPerBlock);
  void WaitForAllGPU(int expectedDone);
  void StopWorkersGPU();
}
