#ifndef HELPER_H_INCLUDED
#define HELER_H_INCLUDED

#include <cuda_runtime_api.h>
#include <iostream>

#define succeed(ans) { succeeded((ans), __FILE__, __LINE__); }
inline void succeeded(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;;
		throw std::runtime_error("GPUasser failed");
	}
}

#endif