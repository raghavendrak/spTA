#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <chrono>
#include <cstdlib>
#include "COOtoCSR.h"

// Function to handle cuSPARSE errors
void checkCusparseStatus(cusparseStatus_t status, const char* msg) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "cuSPARSE error: " << msg << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to convert a sparse matrix in COO format to a dense matrix (row-major 1D array)
double* convertCOOToDense(int64_t rows, int64_t cols, int64_t nonzeros, int64_t* row_pointers, int64_t* col_indices, double* values) {
    double* denseMatrix = new double[rows * cols]();
    
    int64_t idx = 0, ptr = 0;
    while(idx < rows){
        int64_t row = idx;
        while(ptr < row_pointers[idx+1]) {
            int64_t col = col_indices[ptr];
            denseMatrix[row * cols + col] = values[ptr];
            ptr++;
        }
        idx++;
    }
    
    return denseMatrix;
}

// Function to compare two matrices with a given tolerance
bool compareMatrices(int64_t rows, int64_t cols, double* matrixA, double* matrixB, double tolerance) {
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            double diff = std::fabs(matrixA[i * cols + j] - matrixB[i * cols + j]);
            if (diff > tolerance) {
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                          << "A = " << matrixA[i * cols + j] << ", C = " << matrixB[i * cols + j] 
                          << ", Diff = " << diff << std::endl;
                return false;
            }
        }
    }
    return true;
}

// Function to perform SpMM using cuSPARSE
void spmm_cusparse(int64_t m, int64_t n, int64_t p, int64_t* row_pointers, int64_t* col_indices, double* values, double* B, double* C) {
    cusparseHandle_t handle;
    checkCusparseStatus(cusparseCreate(&handle), "Failed to create cuSPARSE handle.");

    // New cuSPARSE descriptors
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    // Number of non-zero elements
    int64_t nnz = row_pointers[m];

    int64_t *d_row_pointers, *d_col_indices;
    double *d_values, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc((void**)&d_row_pointers, (m + 1) * sizeof(int64_t));
    cudaMalloc((void**)&d_col_indices, nnz * sizeof(int64_t));
    cudaMalloc((void**)&d_values, nnz * sizeof(double));
    cudaMalloc((void**)&d_B, n * p * sizeof(double));
    cudaMalloc((void**)&d_C, m * p * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_row_pointers, row_pointers, (m + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, col_indices, nnz * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * p * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, m * p * sizeof(double));

    double alpha = 1.0;
    double beta = 0.0;

    // Create matrix descriptors
    checkCusparseStatus(cusparseCreateCsr(&matA, m, n, nnz,
                                          d_row_pointers, d_col_indices, d_values,
                                          CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
                        "Failed to create CSR matrix descriptor for A.");

    checkCusparseStatus(cusparseCreateDnMat(&matB, n, p, p, d_B, CUDA_R_64F, CUSPARSE_ORDER_ROW),
                        "Failed to create dense matrix descriptor for B.");

    checkCusparseStatus(cusparseCreateDnMat(&matC, m, p, p, d_C, CUDA_R_64F, CUSPARSE_ORDER_ROW),
                        "Failed to create dense matrix descriptor for C.");

    // Buffer size and workspace allocation
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    checkCusparseStatus(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, matA, matB, &beta, matC, CUDA_R_64F,
                                            CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize),
                    "Failed to get buffer size.");


    cudaMalloc(&dBuffer, bufferSize);

    // Perform the SpMM operation
    auto start_gpu = std::chrono::high_resolution_clock::now();

    checkCusparseStatus(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha, matA, matB, &beta, matC, CUDA_R_64F,
                                     CUSPARSE_SPMM_ALG_DEFAULT, dBuffer),
                        "SpMM computation failed.");

    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

    std::cout << "cuSPARSE SpMM execution time: " << gpu_time.count() << " seconds." << std::endl;

    // Copy result back to host
    cudaMemcpy(C, d_C, m * p * sizeof(double), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_row_pointers);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(dBuffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);
}


/*
Command to compile this program : 
nvcc -o spMM_cuSPARSE spMM_cuSPARSE.cu COOtoCSR.cpp -lcusparse

Command to run the program : 
./spMM_cuSPARSE bloweybl.mtx 30003 

*/

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

    // Perform SpMM using cuSPARSE
    spmm_cusparse(A_rows, A_cols, p, row_pointers, col_indices, values, B, C);

    // Convert COO to dense matrix (1D row-major format) for comparison
    double* dense_A = convertCOOToDense(A_rows, A_cols, A_nonzeros, row_pointers, col_indices, values);

    double tolerance = 1e-6;
    if (compareMatrices(A_rows, p, dense_A, C, tolerance)) {
        std::cout << "Matrices A and C are equal within the tolerance of " << tolerance << "." << std::endl;
    } else {
        std::cout << "Matrices A and C are NOT equal within the tolerance of " << tolerance << "." << std::endl;
    }

    // Clean up dynamically allocated memory
    delete[] B;
    delete[] C;
    delete[] row_pointers;
    delete[] col_indices;
    delete[] values;

    return 0;
}
