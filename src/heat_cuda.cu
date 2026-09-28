#include "heat_cuda.hpp"

#include <cstdio>
#include <cstdlib>
#include <utility>

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),   \
                         __FILE__, __LINE__);                                            \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

namespace {

// One GPU thread per interior grid point: the same 5-point update as the CPU versions.
template <typename T>
__global__ void step_kernel(const T* __restrict__ u, T* __restrict__ v, int n, T r) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x + 1;  // column, contiguous in memory
    const int i = blockIdx.y * blockDim.y + threadIdx.y + 1;  // row
    if (i > n || j > n) return;
    const int w = n + 2;
    const int k = i * w + j;
    v[k] = u[k] + r * (u[k - w] + u[k + w] + u[k - 1] + u[k + 1] - T(4) * u[k]);
}

}  // namespace

template <typename T>
double run_cuda(std::vector<T>& u, int n, long steps, T r) {
    const size_t bytes = u.size() * sizeof(T);
    T* d_u = nullptr;
    T* d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_u, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    // Both buffers start as the initial grid so that the boundary ring (always 0) is set in each.
    CUDA_CHECK(cudaMemcpy(d_u, u.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, u.data(), bytes, cudaMemcpyHostToDevice));

    // 32 threads along a row so that neighbouring threads read neighbouring memory.
    const dim3 block(32, 8);
    const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (long s = 0; s < steps; ++s) {
        step_kernel<<<grid, block>>>(d_u, d_v, n, r);
        std::swap(d_u, d_v);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(u.data(), d_u, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_v));
    return ms / 1000.0;
}

template double run_cuda<float>(std::vector<float>&, int, long, float);
template double run_cuda<double>(std::vector<double>&, int, long, double);
