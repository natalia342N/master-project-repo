Requires CMake 3.12 and support for C++14 or newer (C++ and CUDA).
Tested with Turing GPUs, running Windows 10 with CUDA 11, CC 5.2 and 7.5, 64bit and 32bit.

This code is a simplified version of the original evaluation framework. Due to unstable behavior, especially on Turing architectures, most other complex queuing techniques are currently not included for comparison. We will try to add any stable variant that can handle underflow and overflow reliably. If you find any inconsistencies or issues with the code, please feel free to contact us and suggest improvements.

01-03-2020: Please hang tight while we fix issues with the hosted code base!

26-07-2020: We have simplified our original codebase and updated it for modern C++. Initial testing with Turing successful. The framework currently includes code for the Broker Work Distributor (BWD), the Broker Queue (BQ) and the Gottlieb queue. We will be working to add more queuing variants (e.g. LCRQ) as soon as we have established that they are stable on Turing. 