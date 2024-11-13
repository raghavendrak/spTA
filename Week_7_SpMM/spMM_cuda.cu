#include <iostream>
#include <cuda_runtime.h>
#include "COOtoCSR.h"

// // CUDA kernel for SpMM (Sparse Matrix * Dense Matrix)
// __global__ void spmmKernel(int *row_pointers, int *col_indices, double *values,
//                            double *B, double *C, int m, int n, int p) {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     if (row < m) {
//         for (int idx = row_pointers[row]; idx < row_pointers[row + 1]; ++idx) {
//             int col = col_indices[idx];
//             double val = values[idx];
//             for (int j = 0; j < p; ++j) {
//                 atomicAdd(&C[row * p + j], val * B[col * p + j]);
//             }
//         }
//     }
// }

// // Function to perform SpMM using CUDA
// void spmm(int m, int n, int p) {
//     // Allocate and initialize dense matrix B with 1.0 (row-major format)
//     std::vector<double> B(n * p, 1.0);#include <iostream>


// // CUDA kernel for SpMM (Sparse Matrix * Dense Matrix)
// __global__ void spmmKernel(int *row_pointers, int *col_indices, double *values,
//                            double *B, double *C, int m, int n, int p) {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     if (row < m) {
//         for (int idx = row_pointers[row]; idx < row_pointers[row + 1]; ++idx) {
//             int col = col_indices[idx];
//             double val = values[idx];
//             for (int j = 0; j < p; ++j) {
//                 atomicAdd(&C[row * p + j], val * B[col * p + j]);
//             }
//         }
//     }
// }

// // Function to perform SpMM using CUDA
// void spmm(int m, int n, int p) {
//     // Allocate and initialize dense matrix B with 1.0 (row-major format)
//     std::vector<double> B(n * p, 1.0);
//     std::vector<double> C(m * p, 0.0);

//     // Device memory allocation
//     int *d_row_pointers, *d_col_indices;
//     double *d_values, *d_B, *d_C;

//     cudaMalloc((void**)&d_row_pointers, row_pointers.size() * sizeof(int));
//     cudaMalloc((void**)&d_col_indices, col_indices.size() * sizeof(int));
//     cudaMalloc((void**)&d_values, values.size() * sizeof(double));
//     cudaMalloc((void**)&d_B, B.size() * sizeof(double));
//     cudaMalloc((void**)&d_C, C.size() * sizeof(double));

//     // Copy data to device
//     cudaMemcpy(d_row_pointers, row_pointers.data(), row_pointers.size() * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_col_indices, col_indices.data(), col_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_values, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_C, C.data(), C.size() * sizeof(double), cudaMemcpyHostToDevice);

//     // Launch the CUDA kernel
//     int blockSize = 256;
//     int gridSize = (m + blockSize - 1) / blockSize;
//     spmmKernel<<<gridSize, blockSize>>>(d_row_pointers, d_col_indices, d_values, d_B, d_C, m, n, p);

//     // Copy result back to host
//     cudaMemcpy(C.data(), d_C, C.size() * sizeof(double), cudaMemcpyDeviceToHost);

//     // Display the result (C matrix)
//     std::cout << "Result Matrix C:" << std::endl;
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < p; ++j) {
//             std::cout << C[i * p + j] << " ";
//         }
//         std::cout << std::endl;
//     }

//     // Free device memory
//     cudaFree(d_row_pointers);
//     cudaFree(d_col_indices);
//     cudaFree(d_values);
//     cudaFree(d_B);
//     cudaFree(d_C);
// }



int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <MatrixMarketFile>" << std::endl;
        return 1;
    }

    // Declare arrays to hold CSR data
    int64_t *row_pointers = nullptr;
    int64_t *col_indices = nullptr;
    double *values = nullptr;
    int64_t A_rows, A_cols, A_nonzeros;

    try {
        std::string filename = argv[1];
        convertToCSR(filename, row_pointers, col_indices, values, A_rows, A_cols, A_nonzeros);

        // Perform SpMM
        // spmm(A_rows, A_cols, 4, row_pointers, col_indices, values); // Example with p=4 for testing
        std::cout << "Values: ";
        for (int i=0; i<A_nonzeros; i++) std::cout << values[i] << " ";
        std::cout << "\nColumn Indices: ";
        for (int i=0; i<A_nonzeros; i++) std::cout << col_indices[i] << " ";
        std::cout << "\nRow Pointers: ";
        for (int i=0; i<=A_rows; i++) std::cout << row_pointers[i] << " ";
        std::cout << std::endl;
    } catch (const std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    // Clean up dynamically allocated memory
    delete[] row_pointers;
    delete[] col_indices;
    delete[] values;

    return 0;
}