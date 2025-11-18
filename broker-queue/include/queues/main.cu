#include "taskmanager_gpu.cuh"
#include "job_structs.cuh"

using namespace ASC_HPC;

int main()
{
  // 1. Prepare some jobs on the host
  const int numJobs = 16;
  Job jobs[numJobs];

  for (int i = 0; i < numJobs; ++i) {
    jobs[i].type  = JobType::DAXPY;
    jobs[i].jobId = i;
    // fill in device pointers / sizes etc.
  }

  // 2. Start workers (persistent kernel)
  StartWorkersGPU(/*blocks=*/80, /*threadsPerBlock=*/128);

  // 3. Enqueue the jobs
  EnqueueJobsGPU(jobs, numJobs);

  // 4. Wait until all jobs processed
  WaitForAllGPU(numJobs);

  // 5. Stop workers
  StopWorkersGPU();

  return 0;
}
