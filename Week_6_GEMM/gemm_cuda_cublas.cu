#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>  // For atoi
#include <chrono>   // For timing

// Host function to perform GEMM using cuBLAS
void matrixMul(double* A, double* B, double* C, int64_t m, int64_t n, int64_t k) {
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * n * sizeof(double));
    cudaMalloc((void**)&d_B, n * k * sizeof(double));
    cudaMalloc((void**)&d_C, m * k * sizeof(double));
    
    cudaMemcpy(d_A, A, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(double), cudaMemcpyHostToDevice);

    // Set up cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set alpha and beta for the GEMM operation
    double alpha = 1.0;
    double beta = 0.0;

    // Perform matrix multiplication using cublasDgemm
    // C = alpha * A * B + beta * C
    // Note: cuBLAS uses column-major order, so we need to transpose
    //       the order of matrices by adjusting the parameters accordingly
    auto start_gpu = std::chrono::high_resolution_clock::now();
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha, d_A, n, d_B, k, &beta, d_C, m);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

    std::cout << "GPU execution time using cuBLASD : " << gpu_time.count() << " seconds." << std::endl;

    // Copy the result back to host
    cudaMemcpy(C, d_C, m * k * sizeof(double), cudaMemcpyDeviceToHost);

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


// Function to compare two matrices A and C (allowing for floating point tolerance)
bool compareMatrices(double* A, double* C, int64_t m, int64_t k, double tolerance = 1e-6) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            if (std::fabs(A[i * k + j] - C[i * k + j]) > tolerance) {
                std::cout << "Mismatch at A[" << i << "][" << j << "] = " << A[i * k + j] << ", C[" << i << "][" << j << "] = " << C[i * k + j] << std::endl;
                return false;
            }
        }
    }
    return true;
}


/*
Command to compile this program : 
nvcc -o gemm_cuda_cublas gemm_cuda_cublas.cu -lcublas

Command to run the program : 
./gemm_cuda_cublas 30000 30000 30000
*/
int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <m> <n> <k>" << std::endl;
        return -1;
    }

    int64_t m = std::atoi(argv[1]);
    int64_t n = std::atoi(argv[2]);
    int64_t k = std::atoi(argv[3]);

    double* A = new double[m * n];
    double* B = new double[n * k];
    double* C_gpu = new double[m * k];

    // Initialize matrix A and B with some values
    for (int i = 0; i < m * n; i++) {
        A[i] = static_cast<double>(i + 1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            B[i * k + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    matrixMul(A, B, C_gpu, m, n, k);

    // Check if A and C are the same (since B is identity, C should be equal to A)
    if (compareMatrices(A, C_gpu, m, k)) {
        std::cout << "Matrix A and C are the same!" << std::endl;
    } else {
        std::cout << "Matrix A and C are not the same!" << std::endl;
    }

    delete[] A;
    delete[] B;
    delete[] C_gpu;

    return 0;
}
