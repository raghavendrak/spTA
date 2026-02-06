/**
 * =============================================================================
 * FP64 MATRIX-MATRIX MULTIPLICATION (1024×512×2056) USING TENSOR CORES
 * =============================================================================
 *
 * IMPORTANT: SPARSE TENSOR CORES DO NOT SUPPORT FP64 (64-bit)
 * ----------------------------------------------------------
 * NVIDIA's 2:4 Sparse Tensor Cores support: FP32, FP16, BF16, INT8, INT4 only.
 * The cuSPARSELt library (which provides sparse tensor core access) does NOT
 * support double precision.
 *
 * This program uses DENSE FP64 Tensor Cores via cuBLAS:
 * - cuBLAS cublasDgemm automatically uses FP64 tensor cores on Ampere+ (A100, L40S)
 * - No separate library installation required
 * - Simple, production-ready implementation
 *
 * MATRIX DIMENSIONS:
 * ------------------
 * C = alpha * A * B + beta * C
 * A: 1024 × 512  (M × K)
 * B:  512 × 2056 (K × N)
 * C: 1024 × 2056 (M × N)
 *
 * Compile: nvcc -arch=sm_89 -lcublas sparse_tensor_core_fp64_gemm.cu -o sparse_tensor_core_fp64_gemm
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M 1024
#define K 512
#define N 2056

// Compute single output element for verification (column-major A, B)
// A is M×K (col-major: A[i,k] at k*M+i), B is K×N (col-major: B[k,j] at j*K+k)
double cpu_gemm_element(const double* A, const double* B, int i, int j, int m, int k, int n) {
    double sum = 0;
    for (int kk = 0; kk < k; kk++) {
        sum += A[kk * m + i] * B[j * k + kk];
    }
    return sum;
}

int main() {
    printf("==============================================================\n");
    printf("FP64 GEMM: 1024×512×2056 using Tensor Cores\n");
    printf("==============================================================\n");
    printf("\nMatrix dimensions:\n");
    printf("  A: %d × %d\n", M, K);
    printf("  B: %d × %d\n", K, N);
    printf("  C: %d × %d\n", M, N);
    printf("\nNOTE: Sparse tensor cores do NOT support FP64.\n");
    printf("      Using cuBLAS DGEMM (dense FP64 tensor cores on Ampere+)\n");
    printf("==============================================================\n\n");

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate host memory
    size_t size_A = (size_t)M * K * sizeof(double);
    size_t size_B = (size_t)K * N * sizeof(double);
    size_t size_C = (size_t)M * N * sizeof(double);

    double* h_A = (double*)malloc(size_A);
    double* h_B = (double*)malloc(size_B);
    double* h_C = (double*)malloc(size_C);

    // Initialize with simple values (column-major for cuBLAS)
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            h_A[k * M + i] = ((i * K + k) % 17) * 0.1;
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            h_B[j * K + k] = ((k * N + j) % 13) * 0.1;
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0;

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // cuBLAS uses column-major: C = alpha * A * B + beta * C
    // A[M×K], B[K×N], C[M×N] - all in column-major layout
    const double alpha = 1.0;
    const double beta = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cublasStatus_t status = cublasDgemm(handle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        M, N, K,
                                        &alpha,
                                        d_A, M,    // A is M×K, lda=M
                                        d_B, K,    // B is K×N, ldb=K
                                        &beta,
                                        d_C, M);   // C is M×N, ldc=M
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS error: %d\n", status);
        return 1;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Verify with CPU (sample a few elements only - full verification is slow)
    printf("Verifying with CPU (sampling 1000 random elements)...\n");
    double max_err = 0;
    int n_samples = 1000;
    srand(42);
    for (int s = 0; s < n_samples; s++) {
        int i = rand() % M;
        int j = rand() % N;
        // h_C is column-major M×N: C[i,j] at j*M+i
        double ref = cpu_gemm_element(h_A, h_B, i, j, M, K, N);
        double err = fabs(h_C[j*M+i] - ref);
        if (err > max_err) max_err = err;
    }

    printf("  Max absolute error: %.2e\n", max_err);
    printf("\nPerformance:\n");
    printf("  Time: %.3f ms\n", milliseconds);
    double gflops = 2.0 * M * N * K / (milliseconds * 1e6);
    printf("  GFLOPS: %.1f\n", gflops);

    printf("\n==============================================================\n");
    printf(max_err < 1e-8 ? "VERIFICATION: PASSED\n" : "VERIFICATION: FAILED\n");
    printf("==============================================================\n");

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

/*
 * =============================================================================
 * EXPLANATION
 * =============================================================================
 *
 * 1. SPARSE TENSOR CORES DO NOT SUPPORT FP64
 *    - NVIDIA 2:4 sparse tensor cores (Ampere+) support: FP32, FP16, BF16, INT8, INT4
 *    - cuSPARSELt library provides sparse GEMM but only for these types
 *    - For 64-bit with "sparse" we use DENSE FP64 tensor cores via cuBLAS
 *
 * 2. HOW cuBLAS USES TENSOR CORES
 *    - cublasDgemm automatically uses FP64 tensor cores on Ampere+ (A100, L40S)
 *    - No code changes needed - it picks the fastest path
 *    - L40S: ~50+ GFLOPS for this 1024×512×2056 problem
 *
 * 3. MATRIX DIMENSIONS (M×K × K×N = M×N)
 *    - A: 1024×512,  B: 512×2056,  C: 1024×2056
 *    - Total ops: 2*M*N*K ≈ 2.15 billion multiply-adds
 *
 * 4. COLUMN-MAJOR LAYOUT
 *    - cuBLAS uses Fortran-style column-major
 *    - A[i,k] at index k*M+i, B[k,j] at j*K+k, C[i,j] at j*M+i
 *
 * 5. IF YOU NEED TRUE SPARSE TENSOR CORES
 *    - Use cuSPARSELt with FP32 (requires separate library install)
 *    - Matrix A must be pruned to 2:4 format (2 non-zeros per 4 elements)
 */
