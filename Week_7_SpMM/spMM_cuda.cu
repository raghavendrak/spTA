#include <iostream>
#include <cuda_runtime.h>
#include <chrono>   // For timing
#include <cstdlib>  // For atoi
#include "COOtoCSR.h"


__device__ inline double atomicAddDouble(double* address, double value) {
    unsigned long long* address_as_ulong = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *address_as_ulong;
    unsigned long long assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ulong, assumed, __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


__global__ void spmmKernel(int64_t* row_pointers, int64_t* col_indices, double* values,
                           double* B, double* C, int64_t m, int64_t n, int64_t p) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        for (int64_t idx = row_pointers[row]; idx < row_pointers[row + 1]; ++idx) {
            int64_t col = col_indices[idx];
            double val = values[idx];
            for (int64_t j = 0; j < p; ++j) {
                atomicAddDouble(&C[row * p + j], val * B[col * p + j]);
            }
        }
    }
}


// Function to perform SpMM using CUDA
void spmm(int64_t m, int64_t n, int64_t p, int64_t* row_pointers, int64_t* col_indices, double* values, double* B, double* C) {
    // Device memory allocation
    int64_t *d_row_pointers, *d_col_indices;
    double *d_values, *d_B, *d_C;

    cudaMalloc((void**)&d_row_pointers, (m + 1) * sizeof(int64_t));
    cudaMalloc((void**)&d_col_indices, row_pointers[m] * sizeof(int64_t)); // Use the number of non-zeros
    cudaMalloc((void**)&d_values, row_pointers[m] * sizeof(double)); // Use the number of non-zeros
    cudaMalloc((void**)&d_B, n * p * sizeof(double));
    cudaMalloc((void**)&d_C, m * p * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_row_pointers, row_pointers, (m + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, col_indices, row_pointers[m] * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, row_pointers[m] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * p * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, m * p * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    int blockSize = 256;
    int gridSize = (m + blockSize - 1) / blockSize;

    auto start_gpu = std::chrono::high_resolution_clock::now();
    spmmKernel<<<gridSize, blockSize>>>(d_row_pointers, d_col_indices, d_values, d_B, d_C, m, n, p);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

    std::cout << "Thread block size: " << blockSize << ", SpMM GPU execution time: " << gpu_time.count() << " seconds." << std::endl;

    // Copy result back to host
    cudaMemcpy(C, d_C, m * p * sizeof(double), cudaMemcpyDeviceToHost);

    // Display the result (C matrix)
    std::cout << "Result Matrix C:" << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            std::cout << C[i * p + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_row_pointers);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <MatrixMarketFile> <Number of Columns in Matrix B>" << std::endl;
        return 1;
    }

    // Declare arrays to hold CSR data
    int64_t *row_pointers = nullptr;
    int64_t *col_indices = nullptr;
    double *values = nullptr;
    int64_t A_rows, A_cols, A_nonzeros;

    std::string filename = argv[1];
    int64_t p = std::stoll(argv[2]);
    convertToCSR(filename, row_pointers, col_indices, values, A_rows, A_cols, A_nonzeros);

    // Allocate and initialize dense matrix B with 1.0 (row-major format)
    double *B = new double[A_cols * p];
    for (int i = 0; i < A_cols; ++i) {
        for (int j = 0; j < p; ++j) {
            B[i * p + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Allocate result matrix C and initialize to 0.0
    double *C = new double[A_rows * p]();
    
    // Perform SpMM
    spmm(A_rows, A_cols, p, row_pointers, col_indices, values, B, C);

    // Display CSR data
    // std::cout << "Values: ";
    // for (int i = 0; i < A_nonzeros; i++) std::cout << values[i] << " ";
    // std::cout << "\nColumn Indices: ";
    // for (int i = 0; i < A_nonzeros; i++) std::cout << col_indices[i] << " ";
    // std::cout << "\nRow Pointers: ";
    // for (int i = 0; i <= A_rows; i++) std::cout << row_pointers[i] << " ";
    // std::cout << std::endl;
    
    // Clean up dynamically allocated memory
    delete[] B;
    delete[] C;
    delete[] row_pointers;
    delete[] col_indices;
    delete[] values;

    return 0;
}
