#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>  // For atoi
#include <chrono>   // For timing

// Kernel to perform GEMM: C = A * B
__global__ void matrixMulKernel(double* A, double* B, double* C, int64_t m, int64_t n, int64_t k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < k) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// Host function to set up kernel
void matrixMul(double* A, double* B, double* C, int64_t m, int64_t n, int64_t k, int thread_block_size) {
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * n * sizeof(double));
    cudaMalloc((void**)&d_B, n * k * sizeof(double));
    cudaMalloc((void**)&d_C, m * k * sizeof(double));
    
    cudaMemcpy(d_A, A, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(thread_block_size, thread_block_size);
    dim3 gridDim((k + thread_block_size - 1) / thread_block_size, (m + thread_block_size - 1) / thread_block_size);

    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);
    
    cudaMemcpy(C, d_C, m * k * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <m> <n> <k> <thread_block_size>" << std::endl;
        return -1;
    }

    int64_t m = std::atoi(argv[1]);
    int64_t n = std::atoi(argv[2]);
    int64_t k = std::atoi(argv[3]);
    int thread_block_size = std::atoi(argv[4]);

    double* A = new double[m * n];
    double* B = new double[n * k];
    double* C_gpu = new double[m * k];

    for (int i = 0; i < m * n; i++) {
        A[i] = static_cast<double>(i + 1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            B[i * k + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Measure GPU execution time
    auto start_gpu = std::chrono::high_resolution_clock::now();
    matrixMul(A, B, C_gpu, m, n, k, thread_block_size);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

    std::cout << "Thread block size: " << thread_block_size << ", GPU execution time: " << gpu_time.count() << " seconds." << std::endl;

    delete[] A;
    delete[] B;
    delete[] C_gpu;

    return 0;
}
