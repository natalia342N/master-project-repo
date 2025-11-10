#ifndef INCLUDED_SUPPORTED_DATATYPES
#define INCLUDED_SUPPORTED_DATATYPES

class LargerStruct
{
public:
	static constexpr int SIZE = 32;

	__device__ LargerStruct(int x)
	{
		for (int i = 0; i < SIZE; ++i)
		{
			dataA[i] = x;
			dataB[i] = SIZE - i;
		}
	}

	__device__ LargerStruct()
	{
		for (int i = 0; i < SIZE; ++i)
		{
			dataA[i] = 0;
			dataB[i] = 0;
		}
	}

	__device__ int dot() const
	{
		int result = 0;
		for (int i = 0; i < SIZE; ++i)
		{
			result += dataA[i] * dataB[i];
		}
		return result;
	}

private:
	int dataA[SIZE];
	int dataB[SIZE];
};

#endif

// datatypes.h or a new header like daxpytask.h
#ifndef INCLUDED_DAXPY_TASK
#define INCLUDED_DAXPY_TASK

class DaxpyTask
{
public:
    static constexpr int SIZE = 1024; // 3 * 1024 * 4 bytes = 12 KB

    float a;
    float x[SIZE];
    float y[SIZE];

    __device__ DaxpyTask(float a_val)
    {
        a = a_val;
        for (int i = 0; i < SIZE; ++i)
        {
            x[i] = i * 0.5f;
            y[i] = i * 1.0f;
        }
    }

    __device__ DaxpyTask()
    {
        a = 2.0f;
        for (int i = 0; i < SIZE; ++i)
        {
            x[i] = 0.0f;
            y[i] = 0.0f;
        }
    }

    // Original single-threaded version
    __device__ void compute()
    {
        for (int i = 0; i < SIZE; ++i)
        {
            y[i] = a * x[i] + y[i];
        }
    }

    // ✅ NEW block-parallel version
    __device__ void execute_parallel(int tid, int stride)
    {
        for (int i = tid; i < SIZE; i += stride)
        {
            y[i] = a * x[i] + y[i];
        }
    }
};

#endif