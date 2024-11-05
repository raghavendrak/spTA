#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>  // For atoi

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
void matrixMul(double* A, double* B, double* C, int64_t m, int64_t n, int64_t k) {
    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * n * sizeof(double));
    cudaMalloc((void**)&d_B, n * k * sizeof(double));
    cudaMalloc((void**)&d_C, m * k * sizeof(double));
    
    // Copy matrices from host to device
    cudaMemcpy(d_A, A, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(double), cudaMemcpyHostToDevice);

    // Define thread block and grid dimensions
    dim3 blockDim(16, 16);  // A 16x16 block of threads
    dim3 gridDim((k + 15) / 16, (m + 15) / 16);

    // Launch kernel
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);
    
    // Copy the result from device to host
    cudaMemcpy(C, d_C, m * k * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


/*
command to compile this program : 
nvcc -o gemm_cuda gemm_cuda.cu


command to execute the program : 
./gemm_cuda <m> <n> <k>

ex : ./gemm_cuda 3 4 5

*/

int main(int argc, char* argv[]) {
    // Check if there are enough command-line arguments
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <m> <n> <k>" << std::endl;
        return -1;
    }

    // Parse command-line arguments
    int64_t m = std::atoi(argv[1]);
    int64_t n = std::atoi(argv[2]);
    int64_t k = std::atoi(argv[3]);

    // Allocate host matrices (row-major order)
    double* A = new double[m * n];
    double* B = new double[n * k];
    double* C = new double[m * k];

    // Fill matrix A with sequential values (1, 2, 3, ...)
    for (int i = 0; i < m * n; i++) {
        A[i] = static_cast<double>(i + 1);
    }

    std::cout << "matrix A :" << std::endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    // Fill matrix B as an identity matrix (1 on the diagonal)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            B[i * k + j] = (i == j) ? 1.0 : 0.0;  // Use 1.0 instead of 1.0d
        }
    }

    std::cout << "matrix B :" << std::endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            std::cout << B[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    // Call matrix multiplication function
    matrixMul(A, B, C, m, n, k);

    // Print the result
    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            std::cout << C[i * k + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free dynamically allocated memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
