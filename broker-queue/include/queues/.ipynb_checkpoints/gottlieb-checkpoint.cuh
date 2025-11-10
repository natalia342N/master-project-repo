#ifndef INCLUDED_QUEUE_GOTTLIEB
#define INCLUDED_QUEUE_GOTTLIEB

#pragma once

namespace
{
	template <typename D>
	__device__ D atomicLoad(D* data)
	{
		__trap();
	}

	template <>
	__device__ int atomicLoad(int* data)
	{
		return atomicAdd(data, 0);
	}
}

template<unsigned int QUEUE_SIZE, typename T, bool RARE_FULL_OPTIMIZATION=false>
class GottliebQueue
{
	//static_assert(static_popcnt<QUEUE_SIZE>::value == 1, "QUEUE_SIZE must be a power of two!");

private:

	unsigned int head, tail;
	int qi, qu;

#if __CUDA_ARCH__ < 700
	__device__ __forceinline__ void backoff()
	{
		__threadfence();
	}
#else
	__device__ __forceinline__ void backoff()
	{
		__nanosleep(1);
	}
#endif

	volatile unsigned int ids[QUEUE_SIZE];
	T storage[QUEUE_SIZE];

	__device__ bool tir()
	{
		int v = atomicLoad(&qu);
		while (true)
		{
			if (v >= QUEUE_SIZE)
			{
				return false;
			}
			if (atomicAdd(&qu, 1) < QUEUE_SIZE)
			{
				return true;
			}
			v = atomicSub(&qu, 1) - 1;
		}
	}

	__device__ bool tdr()
	{
		int v = atomicLoad(&qi);
		while (true)
		{
			if (v <= 0)
			{
				return false;
			}
			if (atomicSub(&qi, 1) > 0)
			{
				return true;
			}
			v = atomicAdd(&qi, 1) + 1;
		}
	}

public:

	__device__ void init()
	{
		int lid = threadIdx.x + blockIdx.x * blockDim.x;
		for (int i = lid; i < QUEUE_SIZE; i += blockDim.x * gridDim.x)
		{
			ids[i] = 0u;
		}
		if (lid == 0)
		{
			head = tail = 0;
			qi = qu = 0;
		}
	}

	__device__ inline bool enqueue(const T& data)
	{
		if (!tir())
		{
			return false;
		}

		unsigned int ticket = atomicAdd(&tail, 1);
		unsigned int target = ticket % QUEUE_SIZE;
		unsigned int targetid = 2 * (ticket / QUEUE_SIZE);

		while (ids[target] != targetid)
		{
			backoff();
		}

		storage[target] = data;
		__threadfence();
		ids[target] = targetid + 1;

		atomicAdd(&qi, 1);
		return true;
	}

	__device__ inline void dequeue(bool& hasData, T& data)
	{
		hasData = false;
		if (!tdr())
		{
			return;
		}

		hasData = true;

		unsigned int ticket = atomicAdd(&head, 1);
		unsigned int target = ticket % QUEUE_SIZE;
		unsigned int targetid = 2 * (ticket / QUEUE_SIZE) + 1;

		while (ids[target] != targetid)
		{
			backoff();
		}

		data = storage[target];
		__threadfence();
		ids[target] = targetid + 1;

		atomicSub(&qu, 1);
	}
};

#endif  // INCLUDED_QUEUE_GOTTLIEB
