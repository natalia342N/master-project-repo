#include "helper.h"
#include "queuetest.cuh"
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include "datatypes.h"
#include "kernels.cuh"
#include "queues/broker.cuh"
#include "queues/bwd.cuh"
#include "queues/gottlieb.cuh"

// Sets the desired device and prepares the thread counts to use according to the configuration
void QueueTest::prepare(const QueueTest::TestConfiguration config)
{
	// Pick device according to ID and reset
	succeed(cudaSetDevice(config.deviceID));
	succeed(cudaDeviceReset());

	// Print name to check
	succeed(cudaGetDeviceProperties(&props, config.deviceID));
	std::cout << "Running on GPU " << props.name << '\n';

	// Set up the thread counts to test the queue with
	numThreads.clear();
	const int maxThreads = std::min(100000, config.blockSize * props.multiProcessorCount * 8);
	for (int threads = 1; threads <= maxThreads;)
	{
		numThreads.push_back(threads);
		if (threads < config.maxStepSize)
		{
			threads *= 2;
		}
		else
		{
			threads += config.maxStepSize;
		}
	}
	std::cout << "Test will increase thread counts from 1 to " << maxThreads << std::endl;
}

template<typename T, typename Q>
float QueueTest::runTest(int numBlocks, int blockSize, const QueueTest::TestConfiguration cfg)
{
	// Make a clean queue for each run, deallocate everything that could be left over
	succeed(cudaDeviceReset());
	succeed(cudaDeviceSynchronize());

	// Create timing events
	succeed(cudaEventCreate(&start));
	succeed(cudaEventCreate(&end));

	// Determine size of an instance of the selected queue on the device
	unsigned int size;
	unsigned int* dev_Size;
	succeed(cudaMalloc((void**)&dev_Size, sizeof(unsigned int)));
	sizeOnDevice<Q> << <1, 1 >> > (dev_Size);
	succeed(cudaPeekAtLastError());
	succeed(cudaDeviceSynchronize());
	succeed(cudaMemcpy(&size, dev_Size, sizeof(unsigned int), cudaMemcpyDeviceToHost));

	// Allocate memory and use placement new to create on device
	Q* dev_Queue;
	succeed(cudaMalloc((void**)&dev_Queue, (size_t)size));

	placeQueue<Q> << <1, 1 >> > (dev_Queue);
	succeed(cudaPeekAtLastError());
	succeed(cudaDeviceSynchronize());

	// Initiate queue structure
	initTest<Q> << <numBlocks, blockSize >> > (dev_Queue);
	succeed(cudaPeekAtLastError());
	succeed(cudaDeviceSynchronize());

	// Run actual test case
    int* dev_counters;
    succeed(cudaMalloc(&dev_counters, numBlocks * blockSize * sizeof(int)));
    succeed(cudaMemset(dev_counters, 0, numBlocks * blockSize * sizeof(int)));
    
    // Run actual test case
    succeed(cudaEventRecord(start));
    launchTest<Q, T> <<<numBlocks, blockSize>>> (
    	dev_Queue, 
    	cfg.loopIterations, 
    	cfg.enqueueProbability, 
    	cfg.dequeueProbability, 
    	cfg.waitForData,
    	dev_counters);


	succeed(cudaEventRecord(end));
	succeed(cudaEventSynchronize(end));

	// Check for errors
	succeed(cudaPeekAtLastError());
	succeed(cudaDeviceSynchronize());

	// Clean up allocations
	succeed(cudaFree(dev_Size));
	succeed(cudaFree(dev_Queue));

	// Copy results back from device
    float time;
    cudaEventElapsedTime(&time, start, end);
    std::vector<int> counters(static_cast<size_t>(numBlocks * blockSize));
    succeed(cudaMemcpy(counters.data(), dev_counters, counters.size() * sizeof(int), cudaMemcpyDeviceToHost));
    
    int totalTasks = 0;
    for (int val : counters) totalTasks += val;
    
    float tasksPerSecond = totalTasks / (time * 1e-3f);
    std::cout << "Processed tasks: " << totalTasks << ", Throughput: " << tasksPerSecond << " tasks/sec" << std::endl;
    
    // Clean up
    cudaFree(dev_counters);
    
    return time;
}

bool QueueTest::sanityCheck(const QueueTest::TestConfiguration config)
{
	if (config.dequeueProbability > config.enqueueProbability && config.waitForData)
	{
		std::cerr << "Warning: combining higher dequeue probability with waiting can result in deadlock" << std::endl;
		return false;
	}

	return true;
}

template <typename T>
void QueueTest::runTests(const QueueTest::TestConfiguration config)
{
	// Checks the passed test configuration for inconsistencies
	if (!sanityCheck(config))
	{
		std::cerr << "Configuration did not pass sanity check. Press any key to run anyway" << std::endl;
		std::cin.ignore();
	}

	// Selectes device and prepares meta parameters from config
	prepare(config);

	std::cout << "\nLaunching >>" << config.title << "<<" << std::endl;
	std::cout << "Measuring average time over " << config.timingIterations << " iterations for each setup (ms):" << std::endl;

	std::ofstream outfile(config.output + ".csv");
	outfile << "GPU;Threads;ElemSize;Time\n";

	// Run test for all relevant thread counts
	for (size_t t = 0; t < numThreads.size(); t++)
	{
		std::cout << ">> " << t << "/" << numThreads.size() << ", #threads=" << numThreads[t] << ": ";

		float accumTime = 0.0f;
		int blockSize = std::min(numThreads[t], config.blockSize);
		int numBlocks = numThreads[t] / blockSize;

		// Compute time as average over multiple iterations
		for (int i = 0; i < config.timingIterations ;i++)
		{
			// Select appropriate queue
			switch (config.queue)
			{
			case Queue::BROKER_QUEUE:
				accumTime += runTest <T, BrokerQueue < 1<< 18, T, 100000>> (numBlocks, blockSize, config);
				break;
			case Queue::BROKER_WORK_DISTRIBUTOR:
				accumTime += runTest <T, BrokerWorkDistributor < 1 << 18, T, 100000>>(numBlocks, blockSize, config);
				break;
			case Queue::GOTTLIEB_QUEUE:
				accumTime += runTest <T, GottliebQueue < 1 << 18, T>>(numBlocks, blockSize, config);
				break;
			default:
				throw std::runtime_error("Unknown queue!");
			}
		}
		std::cout << accumTime / config.timingIterations << std::endl;

		outfile << props.name << ";" << numThreads[t] << ";" << sizeof(T) << ";" << accumTime << "\n";
	}

	cleanup();
}

// Cleanup (nothing to clean up so far)
void QueueTest::cleanup()
{
}

// Add others to support multiple data types
template void QueueTest::runTests<int>(const QueueTest::TestConfiguration config);
template void QueueTest::runTests<LargerStruct>(const QueueTest::TestConfiguration config);
template void QueueTest::runTests<DaxpyTask>(const QueueTest::TestConfiguration config);
