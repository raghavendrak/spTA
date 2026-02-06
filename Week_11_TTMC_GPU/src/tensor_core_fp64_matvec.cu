/**
 * =============================================================================
 * SIMPLE FP64 TENSOR CORE MATRIX-VECTOR MULTIPLICATION
 * =============================================================================
 * 
 * This program demonstrates how to use NVIDIA Tensor Cores for 64-bit (double
 * precision) matrix-vector multiplication.
 * 
 * REQUIREMENTS:
 * - NVIDIA A100 or newer GPU (Ampere architecture with FP64 Tensor Cores)
 * - CUDA 11.0+
 * - Compile with: nvcc -arch=sm_80 tensor_core_fp64_matvec.cu -o tensor_core_fp64_matvec
 * 
 * HOW TENSOR CORES WORK:
 * ----------------------
 * Tensor Cores perform matrix multiply-accumulate operations on small matrix
 * tiles in a single instruction. For FP64:
 * 
 *   D = A × B + C
 * 
 * Where A, B, C, D are small matrix fragments (tiles).
 * 
 * For FP64 on A100:
 * - A is M×K = 8×4
 * - B is K×N = 4×8
 * - C and D are M×N = 8×8
 * 
 * MATRIX-VECTOR AS MATRIX-MATRIX:
 * -------------------------------
 * Since Tensor Cores do matrix-matrix multiplication, we treat the vector
 * as a matrix with N=8 (padded with zeros). In practice, you'd batch multiple
 * vectors together for better efficiency.
 * 
 * WMMA API:
 * ---------
 * We use the WMMA (Warp Matrix Multiply Accumulate) C++ API:
 * 1. Declare fragment objects for A, B, C, D
 * 2. Load data from memory into fragments
 * 3. Perform mma_sync() operation (executes on tensor cores)
 * 4. Store results back to memory
 */

#include <cuda.h>
#include <mma.h>      // WMMA header
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

// =============================================================================
// TENSOR CORE TILE DIMENSIONS FOR FP64
// =============================================================================
// On A100 (sm_80), FP64 tensor cores use 8×8×4 tile shape
const int M = 8;   // Rows of A and C/D
const int N = 8;   // Cols of B and C/D
const int K = 4;   // Cols of A, Rows of B

// =============================================================================
// TENSOR CORE KERNEL
// =============================================================================
/**
 * Performs: result = matrix × vector + bias
 * 
 * For simplicity, this kernel uses a single warp (32 threads) to compute
 * one 8×8 tile multiplication.
 * 
 * In a real application, you would:
 * 1. Tile the matrix into multiple 8×8 blocks
 * 2. Use multiple warps/blocks
 * 3. Accumulate results from multiple K iterations
 */
__global__ void tensor_core_matvec_fp64(
    const double* __restrict__ matrix,    // M×K matrix (8×4)
    const double* __restrict__ vector,    // K×N vector/matrix (4×8)
    double* __restrict__ result,          // M×N output (8×8)
    const double* __restrict__ bias       // M×N bias (optional, can be zeros)
) {
    // =========================================================================
    // STEP 1: Declare WMMA fragments
    // =========================================================================
    // Fragments are the fundamental data structures for tensor core operations.
    // Each fragment holds a portion of a matrix distributed across a warp.
    
    // Fragment for matrix A (8×4, row-major)
    wmma::fragment<wmma::matrix_a, M, N, K, double, wmma::row_major> a_frag;
    
    // Fragment for matrix B (4×8, row-major) - our "vector" expanded to matrix
    wmma::fragment<wmma::matrix_b, M, N, K, double, wmma::row_major> b_frag;
    
    // Fragment for accumulator C (8×8) - initialized with bias or zeros
    wmma::fragment<wmma::accumulator, M, N, K, double> c_frag;
    
    // =========================================================================
    // STEP 2: Load data from global memory into fragments
    // =========================================================================
    // Leading dimension = number of columns (for row-major layout)
    
    // Load matrix A (8×4)
    wmma::load_matrix_sync(a_frag, matrix, K);  // ldm = K = 4
    
    // Load matrix B (4×8) - this is our vector replicated/expanded
    wmma::load_matrix_sync(b_frag, vector, N);  // ldm = N = 8
    
    // Load bias into accumulator (or initialize to zero)
    wmma::load_matrix_sync(c_frag, bias, N, wmma::mem_row_major);  // ldm = N = 8
    
    // =========================================================================
    // STEP 3: Perform tensor core matrix multiply-accumulate
    // =========================================================================
    // This is where the magic happens!
    // A single mma_sync instruction computes: C = A × B + C
    // This executes on the tensor cores, not regular CUDA cores.
    
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // =========================================================================
    // STEP 4: Store result back to global memory
    // =========================================================================
    wmma::store_matrix_sync(result, c_frag, N, wmma::mem_row_major);
}

// =============================================================================
// HELPER: Print matrix
// =============================================================================
void print_matrix(const char* name, const double* mat, int rows, int cols) {
    printf("\n%s (%d×%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("  [");
        for (int j = 0; j < cols; j++) {
            printf("%8.3f", mat[i * cols + j]);
            if (j < cols - 1) printf(", ");
        }
        printf("]\n");
    }
}

// =============================================================================
// HELPER: CPU reference implementation for verification
// =============================================================================
void cpu_matvec_reference(
    const double* A, const double* B, const double* C, double* D,
    int m, int n, int k
) {
    // D = A × B + C
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = C[i * n + j];
            for (int kk = 0; kk < k; kk++) {
                sum += A[i * k + kk] * B[kk * n + j];
            }
            D[i * n + j] = sum;
        }
    }
}

// =============================================================================
// MAIN
// =============================================================================
int main() {
    printf("==========================================================\n");
    printf("FP64 TENSOR CORE MATRIX-VECTOR MULTIPLICATION DEMO\n");
    printf("==========================================================\n");
    printf("\nTensor Core Tile Dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Operation: Result[%d×%d] = Matrix[%d×%d] × Vector[%d×%d] + Bias\n", 
           M, N, M, K, K, N);
    
    // =========================================================================
    // Check for compatible GPU
    // =========================================================================
    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);
    
    printf("\nGPU: %s (Compute Capability: %d.%d)\n", 
           props.name, props.major, props.minor);
    
    if (props.major < 8) {
        printf("\n*** WARNING: FP64 Tensor Cores require compute capability 8.0+ ***\n");
        printf("*** (A100 or newer GPU). This demo may not work correctly. ***\n");
    }
    
    // =========================================================================
    // Allocate host memory
    // =========================================================================
    size_t size_A = M * K * sizeof(double);  // 8×4 = 32 doubles
    size_t size_B = K * N * sizeof(double);  // 4×8 = 32 doubles  
    size_t size_C = M * N * sizeof(double);  // 8×8 = 64 doubles
    
    double* h_matrix = (double*)malloc(size_A);
    double* h_vector = (double*)malloc(size_B);
    double* h_bias   = (double*)malloc(size_C);
    double* h_result = (double*)malloc(size_C);
    double* h_ref    = (double*)malloc(size_C);
    
    // =========================================================================
    // Initialize test data
    // =========================================================================
    printf("\n--- Initializing Test Data ---\n");
    
    // Matrix A (8×4): Simple sequential values
    for (int i = 0; i < M * K; i++) {
        h_matrix[i] = (double)(i + 1);
    }
    
    // Vector/Matrix B (4×8): For true matrix-vector, first column is our vector
    // Other columns are zeros (or you could batch multiple vectors)
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            if (j == 0) {
                h_vector[i * N + j] = (double)(i + 1);  // Actual vector values
            } else {
                h_vector[i * N + j] = 0.0;  // Padding
            }
        }
    }
    
    // Bias C (8×8): Initialize to zeros
    for (int i = 0; i < M * N; i++) {
        h_bias[i] = 0.0;
    }
    
    print_matrix("Matrix A", h_matrix, M, K);
    print_matrix("Vector B (padded to matrix)", h_vector, K, N);
    
    // =========================================================================
    // Allocate device memory
    // =========================================================================
    double *d_matrix, *d_vector, *d_bias, *d_result;
    cudaMalloc(&d_matrix, size_A);
    cudaMalloc(&d_vector, size_B);
    cudaMalloc(&d_bias,   size_C);
    cudaMalloc(&d_result, size_C);
    
    // =========================================================================
    // Copy data to device
    // =========================================================================
    cudaMemcpy(d_matrix, h_matrix, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias,   h_bias,   size_C, cudaMemcpyHostToDevice);
    
    // =========================================================================
    // Launch kernel
    // =========================================================================
    printf("\n--- Launching Tensor Core Kernel ---\n");
    printf("Grid: 1 block, Block: 32 threads (1 warp)\n");
    
    // IMPORTANT: Tensor Core operations require at least one full warp (32 threads)
    // The WMMA operations are inherently warp-synchronous
    dim3 grid(1);
    dim3 block(32);  // One warp
    
    tensor_core_matvec_fp64<<<grid, block>>>(d_matrix, d_vector, d_result, d_bias);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaDeviceSynchronize();
    
    // =========================================================================
    // Copy result back to host
    // =========================================================================
    cudaMemcpy(h_result, d_result, size_C, cudaMemcpyDeviceToHost);
    
    // =========================================================================
    // Compute CPU reference
    // =========================================================================
    cpu_matvec_reference(h_matrix, h_vector, h_bias, h_ref, M, N, K);
    
    // =========================================================================
    // Display and verify results
    // =========================================================================
    print_matrix("Result (Tensor Cores)", h_result, M, N);
    print_matrix("Reference (CPU)", h_ref, M, N);
    
    // For matrix-vector, we only care about the first column
    printf("\n--- Matrix-Vector Result (first column) ---\n");
    printf("Tensor Core Result vs CPU Reference:\n");
    double max_error = 0.0;
    for (int i = 0; i < M; i++) {
        double tc_val = h_result[i * N + 0];  // First column of result
        double ref_val = h_ref[i * N + 0];
        double error = fabs(tc_val - ref_val);
        max_error = (error > max_error) ? error : max_error;
        printf("  Row %d: TC=%.3f, CPU=%.3f, Error=%.6f\n", i, tc_val, ref_val, error);
    }
    
    printf("\n==========================================================\n");
    printf("Maximum Error: %.10f\n", max_error);
    if (max_error < 1e-10) {
        printf("VERIFICATION: PASSED!\n");
    } else {
        printf("VERIFICATION: FAILED (error too large)\n");
    }
    printf("==========================================================\n");
    
    // =========================================================================
    // Cleanup
    // =========================================================================
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_bias);
    cudaFree(d_result);
    free(h_matrix);
    free(h_vector);
    free(h_bias);
    free(h_result);
    free(h_ref);
    
    return 0;
}

/**
 * =============================================================================
 * EXPLANATION SUMMARY
 * =============================================================================
 * 
 * 1. WHAT ARE TENSOR CORES?
 *    - Specialized hardware units in NVIDIA GPUs (Volta and newer)
 *    - Perform matrix multiply-accumulate in a single instruction
 *    - FP64 tensor cores available only on A100+ (sm_80+)
 * 
 * 2. WMMA API WORKFLOW:
 *    a) Declare fragments (distributed matrix portions across a warp)
 *    b) Load data from memory → fragments (load_matrix_sync)
 *    c) Execute D = A × B + C (mma_sync) 
 *    d) Store results fragment → memory (store_matrix_sync)
 * 
 * 3. KEY CONCEPTS:
 *    - Operations are WARP-SYNCHRONOUS (all 32 threads cooperate)
 *    - Data is distributed across threads in the warp
 *    - Fixed tile sizes: 8×4 × 4×8 → 8×8 for FP64
 * 
 * 4. MATRIX-VECTOR ADAPTATION:
 *    - Tensor cores do matrix-matrix, not matrix-vector directly
 *    - We treat vector as Kx1 matrix, pad to KxN for tensor core shape
 *    - First column of result contains the matrix-vector product
 * 
 * 5. COMPILATION:
 *    nvcc -arch=sm_80 tensor_core_fp64_matvec.cu -o tensor_core_fp64_matvec
 * 
 * 6. PERFORMANCE NOTE:
 *    - This demo is for learning, not optimal performance
 *    - Real applications tile large matrices and use multiple warps/blocks
 *    - A100 achieves ~19.5 TFLOPS with FP64 tensor cores
 */
