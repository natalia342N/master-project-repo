#include <iostream>
#include "queuetest.cuh"
#include <string>
#include <stdexcept>
#include "datatypes.h"

// Pass device ID for GPU on which test should be run as first command line parameter
int main(int argc, char* argv[])
{
	QueueTest::TestConfiguration config;

	if (argc > 1)
	{
		std::string arg(argv[1]);
		try 
		{
			std::size_t pos;
			config.deviceID = std::stoi(arg, &pos);
			if (pos < arg.size()) 
			{
				std::cerr << "Trailing characters after device ID: " << arg << '\n';
			}
		}
		catch (std::exception const& ) 
		{
			std::cerr << "Usage: " << arg[0] << " <device ID>" << '\n';
		}
	}

	using TestSetup = QueueTest::TestConfiguration;
	using QueueType = QueueTest::Queue;

	// Describe individual test to run
	TestSetup basicBWD = config;
	basicBWD.title = "Basic Test (BWD)";
	basicBWD.output = "bwd_basic_test";
	basicBWD.queue = QueueType::BROKER_WORK_DISTRIBUTOR;;

	TestSetup basicGottlieb = config;
	basicGottlieb.title = "Basic Test (Gottlieb)";
	basicGottlieb.output = "gottlieb_basic_test";
	basicGottlieb.queue = QueueType::GOTTLIEB_QUEUE;

	TestSetup lowDeqBWD = config;
	lowDeqBWD.title = "Deq 25% (BWD)";
	lowDeqBWD.output = "bwd_deq_25";
	lowDeqBWD.queue = QueueType::BROKER_WORK_DISTRIBUTOR;
	lowDeqBWD.dequeueProbability = 0.25f;
	lowDeqBWD.waitForData = false;

	TestSetup lowDeqGottlieb = config;
	lowDeqGottlieb.title = "Deq 25% (Gottlieb)";
	lowDeqGottlieb.output = "gottlieb_deq_25";
	lowDeqGottlieb.queue = QueueType::GOTTLIEB_QUEUE;
	lowDeqGottlieb.dequeueProbability = 0.25f;
	lowDeqGottlieb.waitForData = false;

	TestSetup setups[] = { basicBWD, basicGottlieb, lowDeqBWD, lowDeqGottlieb };
	
	QueueTest test;

	// Run test cases, using integers as queued up elements
	// Testing measures runtime. Results are written to stdout and to a .csv
	// named according to .output field.
	for (const TestSetup& setup : setups)
	{
		test.runTests<int>(setup);
	}

	// Can also queue larger structs. Add new types to "supported_datatypes.h"
	// and force instantiation of runTests template in queueTest.cu
	// Data types must support a default constructor and a constructor from
	// a single int.
	for (TestSetup& setup : setups)
	{
		setup.title.append(" Large");
		setup.output.append("_large");
		test.runTests<LargerStruct>(setup);
	}

	std::cout << "All done." << std::endl;
}