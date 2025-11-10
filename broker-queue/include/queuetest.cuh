#ifndef INCLUDED_QUEUE_TEST
#define INCLUDED_QUEUE_TEST

#pragma once

#include <cuda_runtime_api.h>
#include <vector>

/*
QueueTest class

Contains logic for preparing and launching the queue test cases, as well as configuration properties.
*/
class QueueTest
{
public:

	// Currently supported queue types
	enum class Queue { BROKER_WORK_DISTRIBUTOR, BROKER_QUEUE, GOTTLIEB_QUEUE };

	// Configuration for a single test case
	struct TestConfiguration
	{
		// Name to show during the execution of the test case
		std::string title = "Default Test";

		std::string output = "output";

		// The queue to use for the test
		Queue queue = Queue::BROKER_WORK_DISTRIBUTOR;

		// The device ID of the GPU that should be used
		int deviceID = 0;

		// The blocksize configuraiton for the kernel launch
		int blockSize = 256;

		// The maximum step size by which threads are increased (#threads grows exponentially until this number is reached)
		int maxStepSize = 1024;

		// Number of iterations for measuring runtime
		int timingIterations = 3;

		// The number of loop iterations in the kernel
		int loopIterations = 10;

		// Probability of the kernel performing an enqueue in each loop iteration
		float enqueueProbability = 1.0f;

		// Probability of the kernel performing a dequeue in each loop iteration
		float dequeueProbability = 1.0f;

		// Forces the kernel to wait for a value from the dequeue operation (retries until returns non-empty)
		bool waitForData = true;
	};

	// Run tests for all thread counts with the passed configuration
	template <typename T>
	void runTests(const TestConfiguration config);

private:

	cudaDeviceProp props;

	cudaEvent_t start, end;

	std::vector<int> numThreads;

	void prepare(const TestConfiguration config);

	bool sanityCheck(const TestConfiguration config);

	template<typename T, typename Q>
	float runTest(int numBlocks, int blockSize, const QueueTest::TestConfiguration config);

	void cleanup();
};

#endif