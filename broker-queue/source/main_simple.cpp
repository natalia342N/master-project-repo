#include <iostream>
#include "queuetest.cuh"
#include <string>
#include <stdexcept>
#include "datatypes.h"

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
        catch (std::exception const&) 
        {
            std::cerr << "Usage: " << argv[0] << " <device ID>" << '\n';
        }
    }

    // Basic setup for DAXPY test
    config.title = "DAXPY Cooperative Small Test";
    config.output = "broker_daxpy_small";
    config.queue = QueueTest::Queue::BROKER_QUEUE;   // or your preferred queue
    config.dequeueProbability = 1.0f;
    config.enqueueProbability = 1.0f;
    config.waitForData = true;

    config.blockSize = 32;
    config.maxStepSize = 1024;         // Avoid skipping values when growing thread count
    config.timingIterations = 1;    // Fast test loop

    QueueTest().runTests<DaxpyTask>(config);

    std::cout << "Test completed." << std::endl;
}
