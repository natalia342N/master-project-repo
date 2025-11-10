#include <cstdio>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

constexpr int    N         = 1024;       
constexpr double TOL       = 1e-6;      
constexpr int    MAX_ITERS = 1000;        

__global__ void spmv(int n, const double* x, double* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double v = 2.0 * x[i];
    if (i > 0)     v -= x[i-1];
    if (i < n-1)   v -= x[i+1];
    y[i] = v;
}


__global__ void axpy(int n, double* y, const double* x, double alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}

__global__ void axpy_neg(int n, double* r, const double* q, double alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] -= alpha * q[i];
}

__global__ void update_p(int n, double* p, const double* r, double beta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = r[i] + beta * p[i];
}

int main()
{
    double *d_x, *d_b, *d_r, *d_p, *d_q, *d_dot_rr, *d_dot_pq;
    cudaMalloc(&d_x,  N*sizeof(double));
    cudaMalloc(&d_b,  N*sizeof(double));
    cudaMalloc(&d_r,  N*sizeof(double));
    cudaMalloc(&d_p,  N*sizeof(double));
    cudaMalloc(&d_q,  N*sizeof(double));
    cudaMalloc(&d_dot_rr, sizeof(double));
    cudaMalloc(&d_dot_pq, sizeof(double));

    std::vector<double> h_b(N, 1.0);
    cudaMemcpy(d_b, h_b.data(), N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, N*sizeof(double));

    cudaMemcpy(d_r, d_b, N*sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_p, d_b, N*sizeof(double), cudaMemcpyDeviceToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    cublasDdot(handle, N, d_r, 1, d_r, 1, d_dot_rr);
    double res0;
    cudaMemcpy(&res0, d_dot_rr, sizeof(double), cudaMemcpyDeviceToHost);
    double res = res0;

    const dim3 block(256);
    const dim3 grid((N + block.x - 1) / block.x);

    spmv<<<grid, block, 0, stream>>>(N, d_p, d_q);
    cublasDdot(handle, N, d_p, 1, d_q, 1, d_dot_pq);
    double pq;
    cudaMemcpy(&pq, d_dot_pq, sizeof(double), cudaMemcpyDeviceToHost);
    double alpha = res / pq;

    axpy    <<<grid, block, 0, stream>>>(N, d_x, d_p,  alpha);
    axpy_neg<<<grid, block, 0, stream>>>(N, d_r, d_q, alpha);

    cublasDdot(handle, N, d_r, 1, d_r, 1, d_dot_rr);
    cudaMemcpy(&res, d_dot_rr, sizeof(double), cudaMemcpyDeviceToHost);
    double beta = 0.0;
    update_p<<<grid, block, 0, stream>>>(N, d_p, d_r, beta);

    int iter = 1;

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    spmv<<<grid, block, 0, stream>>>(N, d_p, d_q);
    cublasDdot(handle, N, d_p, 1, d_q, 1, d_dot_pq);

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    while (std::sqrt(res / res0) > TOL && iter < MAX_ITERS)
    {
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);

        cudaMemcpy(&pq, d_dot_pq, sizeof(double), cudaMemcpyDeviceToHost);
        alpha = res / pq;
        axpy    <<<grid, block>>>(N, d_x, d_p,  alpha);
        axpy_neg<<<grid, block>>>(N, d_r, d_q, alpha);
        cublasDdot(handle, N, d_r, 1, d_r, 1, d_dot_rr);
        cudaMemcpy(&res, d_dot_rr, sizeof(double), cudaMemcpyDeviceToHost);
        beta = res / (alpha * pq);   // classical formula using previous α · (p,q)
        update_p<<<grid, block>>>(N, d_p, d_r, beta);

        ++iter;
    }

    printf("CG converged in %d iterations, relative residual %.3e\n",
           iter, std::sqrt(res / res0));

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cublasDestroy(handle);

    cudaFree(d_x);     cudaFree(d_b);     cudaFree(d_r);
    cudaFree(d_p);     cudaFree(d_q);
    cudaFree(d_dot_rr); cudaFree(d_dot_pq);
    return 0;
}
