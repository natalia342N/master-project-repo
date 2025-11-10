template <int N>
class DaxpyTaskN {
public:
    float a;
    float x[N];
    float y[N];

    __device__ DaxpyTaskN() {
        a = 2.0f;
        for (int i = 0; i < N; ++i) {
            x[i] = 0.5f * i;
            y[i] = 1.0f * i;
        }
    }

    __device__ void compute() {
        for (int i = 0; i < N; ++i) {
            y[i] = a * x[i] + y[i];
        }
    }

    __device__ void execute_parallel(int tid, int stride) {
        for (int i = tid; i < N; i += stride) {
            y[i] = a * x[i] + y[i];
        }
    }
};
