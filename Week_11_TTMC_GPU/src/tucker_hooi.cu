/**
 * Tucker Decomposition via Higher-Order Orthogonal Iteration (HOOI)
 *
 * Reads a sparse tensor in COO format (.tns) directly, then builds 3 CSF copies
 * in memory (one per mode as root) for optimal compression at each TTMc step.
 * All TTMc operations use the single ncm_0 kernel with the appropriate CSF copy:
 *   - For TTMc skipping mode n, CSF copy n has mode n as root.
 *   - The remaining modes in each copy are sorted ascending by unique-index count
 *     (SPLATT-style compression).
 *
 * Usage:
 * nvcc -O3 -arch=sm_89 ./src/tucker_hooi.cu -o tucker_hooi.out -lcusolver -lcublas -lcudart
 * ./tucker_hooi.out [-v] [-c] -r 50 50 50 -t 0.01 -m 10 ./tns_datasets/nell-2.tns
 *
 * Flags:
 *   -v / --verbose   Print detailed per-step timing and SVD internals.
 *   -c / --check     Run CPU reference TTMc in iteration 0 and report error vs GPU.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <random>
#include <unordered_set>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
// #include "matrix_utils.h"


// ===================================================================
// TTMc GPU Kernel (ncm_0): one block per root fiber.
// Computes arr_O[i * f1*f2 + r*f2 + s] += sum_j sum_k X[i,j,k]*A[j,r]*B[k,s]
// i = mode_0_idx[blockIdx.x]  (non-contracting / root mode index)
// arr_A = factor for level-1 (middle) mode, arr_B = factor for level-2 (leaf) mode.
// Both factors are row-major on GPU: A[row,col] = arr_A[row*f1 + col].
// ===================================================================
__global__ void GPU_4L_CM_device_func_ncm_0(
  const uint64_t* __restrict__ mode_0_idx,
  const uint64_t* __restrict__ mode_1_ptr, const uint64_t* __restrict__ mode_1_idx,
  const uint64_t* __restrict__ mode_2_ptr, const uint64_t* __restrict__ mode_2_idx,
  const float* __restrict__ values, float* arr_A,  float* arr_B,  float* arr_O,
  uint32_t f1, uint32_t f2,  int num_warps)
{
  extern __shared__ float buf[];
  __shared__ int s_counter;
  int buf_index;

  uint64_t i_ptr = blockIdx.x;
  uint64_t i =  mode_0_idx[i_ptr];

  uint32_t warp_size = 32;
  uint32_t warp_id = threadIdx.x / warp_size;
  int tid_in_warp = threadIdx.x % warp_size;

  for(int buf_idx = threadIdx.x; buf_idx < (int)(f1 * f2); buf_idx += blockDim.x)
    buf[num_warps * f2 + buf_idx] = 0.0f;
  if (threadIdx.x == 0) s_counter = 0;
  __syncthreads();

  uint64_t offset, j_ptr, j_ptr_offset = mode_1_ptr[i_ptr];
  unsigned int full_mask = 0xFFFFFFFFu;

  while(true){
    if(tid_in_warp == 0) offset = atomicAdd(&s_counter, 1);
    offset = __shfl_sync(full_mask, offset, 0);
    j_ptr = j_ptr_offset + offset;
    if(j_ptr < mode_1_ptr[i_ptr + 1]){
      uint64_t j = mode_1_idx[j_ptr];

      for(int buf_idx_offset = warp_id * f2; buf_idx_offset < (int)((warp_id + 1)* f2); buf_idx_offset += warp_size){
        buf_index = buf_idx_offset + tid_in_warp;
        if(buf_index < (int)((warp_id + 1)* f2))
          buf[buf_index] = 0.0f;
      }

      for(uint64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr){
        uint64_t k = mode_2_idx[k_ptr];
        for(uint32_t s_offset = 0; s_offset < f2; s_offset += warp_size){
          uint32_t s = s_offset + tid_in_warp;
          if(s < f2)
            buf[warp_id * f2 + s] += values[k_ptr] * arr_B[k * f2 + s];
        }
      }

      for(uint32_t r = 0; r < f1; ++r){
        for(uint32_t s_offset = 0; s_offset < f2; s_offset += warp_size){
          uint32_t s = s_offset + tid_in_warp;
          if(s < f2)
            atomicAdd(&buf[num_warps * f2 + r * f2 + s], buf[warp_id * f2 + s] * arr_A[j * f1 + r]);
        }
      }
    } else {
      break;
    }
  }
  __syncthreads();

  for(uint32_t r_offset = 0; r_offset < f1; r_offset += num_warps){
    uint32_t r = r_offset + warp_id;
    if(r < f1){
      for(uint32_t s_offset = 0; s_offset < f2; s_offset += warp_size){
        uint32_t s = s_offset + tid_in_warp;
        if(s < f2)
          arr_O[i * f1* f2 + r * f2 + s] += buf[num_warps * f2 + r * f2 + s];
      }
    }
  }
}


#define CHECK_CUDA(call)                                                        \
  do {                                                                        \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
           << " \"" << cudaGetErrorString(err) << "\"\n";            \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
  } while (0)

inline const char* cusolverStatusString(cusolverStatus_t s) {
  switch (s) {
    case CUSOLVER_STATUS_SUCCESS:                return "SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:        return "NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:           return "ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:          return "INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:          return "ARCH_MISMATCH";
    case CUSOLVER_STATUS_MAPPING_ERROR:          return "MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED:       return "EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:         return "INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED:          return "NOT_SUPPORTED";
    case CUSOLVER_STATUS_ZERO_PIVOT:             return "ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:        return "INVALID_LICENSE";
    default:                                     return "UNKNOWN";
  }
}

#define CHECK_CUSOLVER(call)                                                    \
  do {                                                                        \
    cusolverStatus_t status = (call);                                       \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                \
      std::cerr << "cuSOLVER error at " << __FILE__ << ":" << __LINE__    \
           << " status=" << cusolverStatusString(status) << "\n";     \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
  } while (0)

#define CHECK_CUBLAS(call)                                                      \
  do {                                                                        \
    cublasStatus_t status = (call);                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                  \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__      \
           << " status=" << static_cast<int>(status) << "\n";        \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
  } while (0)


// ===================================================================
// GPU memory tracking (enabled by -g / --gpu-stats flag)
// Shadows cudaMalloc so every allocation prints its size when the flag is set.
// Must be defined after cuda_runtime.h is included.
// ===================================================================
static bool g_gpu_stats = false;
uint64_t total_gpu_malloc_size = 0;
inline cudaError_t _tracked_cudaMalloc(void** ptr, size_t bytes,
                                       const char* file, int line) {
  if (g_gpu_stats) {
    double mb = bytes / (1024.0 * 1024.0);
    double gb = mb / 1024.0;
    std::cout << "[GPU malloc] " 
              << mb << " MB  (" << gb << " GB)  " << file << ":" << line << "\n";
    total_gpu_malloc_size += bytes;
    std::cout << "[Total GPU malloc] " << total_gpu_malloc_size / (1024.0 * 1024.0 * 1024.0) << " GB\n";
  }
  return cudaMalloc(ptr, bytes);
}
#define cudaMalloc(ptr, size) _tracked_cudaMalloc((void**)(ptr), (size), __FILE__, __LINE__)


// Initialize factor matrix with random orthonormal columns (Gram-Schmidt)
void init_factor_orthonormal(uint64_t rows, uint64_t cols, unsigned int seed, float* A) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (uint64_t i = 0; i < rows * cols; i++) A[i] = dist(gen);
  for (uint64_t c = 0; c < cols; c++) {
    float norm = 0;
    for (uint64_t r = 0; r < rows; r++) norm += A[r * cols + c] * A[r * cols + c];
    norm = std::sqrt(norm);
    if (norm < 1e-10f) norm = 1.0f;
    for (uint64_t r = 0; r < rows; r++) A[r * cols + c] /= norm;
    for (uint64_t c2 = c + 1; c2 < cols; c2++) {
      float dot = 0;
      for (uint64_t r = 0; r < rows; r++) dot += A[r * cols + c] * A[r * cols + c2];
      for (uint64_t r = 0; r < rows; r++) A[r * cols + c2] -= dot * A[r * cols + c];
    }
  }
}

// Function for aligned memory allocation
float* allocate_aligned_array(size_t num_elements) {
  constexpr size_t alignment = 32;           // 32 bytes = 256 bits
  constexpr size_t element_size = sizeof(float); // 8 bytes per float

  size_t total_bytes = num_elements * element_size;

  // Pad to next multiple of 32 bytes if needed
  if (total_bytes % alignment != 0) {
    total_bytes = ((total_bytes + alignment - 1) / alignment) * alignment;
  }

  // Now, allocate aligned memory
  void* ptr = std::aligned_alloc(alignment, total_bytes);
  if (!ptr) {
    throw std::runtime_error("Failed to allocate aligned memory");
  }

  //initilaize to zero
  size_t total_elements = total_bytes / element_size;
  float* arr = static_cast<float*>(ptr);
  for (size_t i = 0; i < total_elements; ++i) {
    arr[i] = 0.0;
  }

  return static_cast<float*>(ptr);
}

// Frobenius norm squared (sum of v^2 over all nnz values)
float frobenius_norm_sq_sparse(const float* values, size_t nnz) {
  double sum = 0;
  for (size_t i = 0; i < nnz; i++) { double v = values[i]; sum += v * v; }
  return static_cast<float>(sum);
}

// Truncated SVD via eigendecomposition of the Gram matrix.
// d_A: M×N col-major. Output: top-R left singular vectors in d_factor (M×R col-major).
// verbose: if false, suppress all internal timing prints.
void gpu_truncated_svd_update_factor(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH,
  float* d_A, int M, int N, int R, float* d_factor, bool verbose) {
  float alpha = 1.0f, beta = 0.0f;
  int K = std::min(M, N);
  R = std::min(R, K);

  cudaEvent_t ev_start, ev_stop;
  float ev_ms = 0.f;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);

  if (M > N) {
    float* d_Gram;
    cudaEventRecord(ev_start);
    CHECK_CUDA(cudaMalloc(&d_Gram, sizeof(float) * N * N));
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  ATA alloc: " << ev_ms << " ms\n";


    cublasMath_t mode;
    cublasGetMathMode(cublasH, &mode);
    std::cout << "Math mode: " << mode << "\n";
    // 0 = CUBLAS_DEFAULT_MATH (TF32 on Ampere+)
    // 1 = CUBLAS_PEDANTIC_MATH (strict FP32)
    
    cudaEventRecord(ev_start);
    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
      N, N, M, &alpha, d_A, M, d_A, M, &beta, d_Gram, N));
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  ATA gemm: " << ev_ms << " ms\n";

    float *d_W;
    CHECK_CUDA(cudaMalloc(&d_W, sizeof(float) * N));
    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnSsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_UPPER, N, d_Gram, N, d_W, &lwork));
    // std::cout << "  lwork (MB): " << lwork / (1024.0 * 1024.0) << "\n";

    float *d_work; int *d_info;
    cudaEventRecord(ev_start);
    CHECK_CUDA(cudaMalloc(&d_work, sizeof(float) * lwork));
    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  eig buf alloc: " << ev_ms << " ms\n";
    std::cout << "  d_work (MB): " << sizeof(float) * lwork / (1024.0 * 1024.0) << "\n";

    cudaEventRecord(ev_start);
    CHECK_CUSOLVER(cusolverDnSsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_UPPER, N, d_Gram, N, d_W, d_work, lwork, d_info));
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  eig decomp: " << ev_ms << " ms\n";

    float* d_V_R = d_Gram + (long long)(N - R) * N;
    cudaEventRecord(ev_start);
    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
      M, R, N, &alpha, d_A, M, d_V_R, N, &beta, d_factor, M));
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  AV gemm: " << ev_ms << " ms\n";

    float* h_W = new float[N];
    CHECK_CUDA(cudaMemcpy(h_W, d_W, sizeof(float) * N, cudaMemcpyDeviceToHost));
    cudaEventRecord(ev_start);
    for (int j = 0; j < R; j++) {
      float sigma = std::sqrt(std::max(h_W[N - R + j], 1e-12f));
      float scale = 1.0f / sigma;
      CHECK_CUBLAS(cublasSscal(cublasH, M, &scale, d_factor + (long long)j * M, 1));
    }
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  normalize (1/sigma): " << ev_ms << " ms\n";
    delete[] h_W;

    CHECK_CUDA(cudaFree(d_Gram)); CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaFree(d_work)); CHECK_CUDA(cudaFree(d_info));
  } else {
    float* d_Gram;
    cudaEventRecord(ev_start);
    CHECK_CUDA(cudaMalloc(&d_Gram, sizeof(float) * M * M));
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  AA^T alloc: " << ev_ms << " ms\n";

    cudaEventRecord(ev_start);
    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
      M, M, N, &alpha, d_A, M, d_A, M, &beta, d_Gram, M));
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  AA^T gemm: " << ev_ms << " ms\n";

    float *d_W;
    CHECK_CUDA(cudaMalloc(&d_W, sizeof(float) * M));
    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnSsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_UPPER, M, d_Gram, M, d_W, &lwork));

    float *d_work; int *d_info;
    cudaEventRecord(ev_start);
    CHECK_CUDA(cudaMalloc(&d_work, sizeof(float) * lwork));
    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  eig buf alloc: " << ev_ms << " ms\n";

    cudaEventRecord(ev_start);
    CHECK_CUSOLVER(cusolverDnSsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_UPPER, M, d_Gram, M, d_W, d_work, lwork, d_info));
    cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ev_ms, ev_start, ev_stop);
    if (verbose) std::cout << "  eig decomp: " << ev_ms << " ms\n";

    CHECK_CUDA(cudaMemcpy(d_factor, d_Gram + (long long)(M - R) * M,
      sizeof(float) * M * R, cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaFree(d_Gram)); CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaFree(d_work)); CHECK_CUDA(cudaFree(d_info));
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);
}


// Full SVD: cusolverDnSgesvd when M > N (typical); eigendecomposition of AA^T
// when M <= N (degenerate small mode — cuSOLVER's Sgesvd doesn't support M<N reliably).
// d_A: M×N col-major (destroyed on return). Output: top-R left singular vectors
// in d_factor (M×R col-major).
void gpu_full_svd_update_factor(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH,
  float* d_A, int M, int N, int R, float* d_factor, bool verbose) {
  int min_mn = std::min(M, N);
  R = std::min(R, min_mn);

  cudaEvent_t ev0, ev1; float ev_ms = 0.f;
  cudaEventCreate(&ev0); cudaEventCreate(&ev1);
  std::cout << "  M = " << M << ", N = " << N << ", R = " << R << "\n";
  std::cout << "MxN matrix size= " << M * N  * sizeof(float) / (1024.0 * 1024.0) << " MB\n";

  if (M > N) {
    // --- cusolverDnSgesvd: jobu='S', jobvt='N' (works when M > N) ---
    float *d_S, *d_U, *d_VT_dummy; int *d_info;
    CHECK_CUDA(cudaMalloc(&d_S,        sizeof(float) * min_mn));
    CHECK_CUDA(cudaMalloc(&d_U,        sizeof(float) * M * min_mn));
    CHECK_CUDA(cudaMalloc(&d_VT_dummy, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_info,     sizeof(int)));
    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(cusolverH, M, N, &lwork));
    float *d_work;
    std::cout << "  d_work (MB): " << sizeof(float) * std::max(lwork, 1) / (1024.0 * 1024.0) << "\n";
    std::cout << "ratio of d_work to MxN matrix size= " << (float) sizeof(float) * std::max(lwork, 1) / (M * N * sizeof(float)) << "\n";
    CHECK_CUDA(cudaMalloc(&d_work, sizeof(float) * std::max(lwork, 1)));


    cublasMath_t mode;
    cublasGetMathMode(cublasH, &mode);
    std::cout << "Math mode: " << mode << "\n";
    // 0 = CUBLAS_DEFAULT_MATH (TF32 on Ampere+)
    // 1 = CUBLAS_PEDANTIC_MATH (strict FP32)

    cudaEventRecord(ev0);
    CHECK_CUSOLVER(cusolverDnSgesvd(cusolverH, 'S', 'N', M, N, d_A, M,
      d_S, d_U, M, d_VT_dummy, 1, d_work, lwork, nullptr, d_info));
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&ev_ms, ev0, ev1);
    if (verbose) std::cout << "  cusolverDnSgesvd: " << ev_ms << " ms\n";

    int h_info = 0;
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0)
      std::cerr << "  WARNING: cusolverDnSgesvd info=" << h_info << "\n";

    // First R columns of d_U are contiguous (col-major, descending order) → d_factor
    CHECK_CUDA(cudaMemcpy(d_factor, d_U, sizeof(float) * M * R, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaFree(d_S)); CHECK_CUDA(cudaFree(d_U));
    CHECK_CUDA(cudaFree(d_VT_dummy)); CHECK_CUDA(cudaFree(d_info));
    CHECK_CUDA(cudaFree(d_work));
  } else {
    // --- M <= N: SVD of A^T (N×M, now tall-skinny N > M) ---
    // d_A is col-major M×N (lda=M). Transpose to col-major N×M (lda=N) via cublasSgeam.
    // A^T = P·S·VT  =>  A = VT^T·S·P^T  =>  left sing vecs of A = cols of VT^T = rows of VT.
    // Use jobvt='S' to get VT (M×M col-major), then extract: d_factor = VT^T[:, 0:R].
    // M is tiny (< N = rank product) so all allocations here are negligible.
    float one = 1.f, zero = 0.f;

    // Step 1: transpose d_A (M×N col-major) → d_At (N×M col-major)
    float *d_At;
    CHECK_CUDA(cudaMalloc(&d_At, sizeof(float) * N * M));
    CHECK_CUBLAS(cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, N, M,
      &one, d_A, M, &zero, d_At, N, d_At, N));

    // Step 2: cusolverDnSgesvd on A^T (m=N > n=M) — jobu='S', jobvt='S'
    float *d_S, *d_P, *d_VT; int *d_info;
    CHECK_CUDA(cudaMalloc(&d_S,    sizeof(float) * M));        // min(N,M)=M singular values
    CHECK_CUDA(cudaMalloc(&d_P,    sizeof(float) * N * M));    // jobu='S': left vecs of A^T (N×M)
    CHECK_CUDA(cudaMalloc(&d_VT,   sizeof(float) * M * M));    // jobvt='S': right vecs of A^T (M×M) = U^T of A
    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(cusolverH, N, M, &lwork));
    float *d_work;
    CHECK_CUDA(cudaMalloc(&d_work, sizeof(float) * std::max(lwork, 1)));
    std::cout << "  d_work (MB): " << sizeof(float) * std::max(lwork, 1) / (1024.0 * 1024.0) << "\n";
    cudaEventRecord(ev0);
    CHECK_CUSOLVER(cusolverDnSgesvd(cusolverH, 'S', 'S', N, M, d_At, N,
      d_S, d_P, N, d_VT, M, d_work, lwork, nullptr, d_info));
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    cudaEventElapsedTime(&ev_ms, ev0, ev1);
    if (verbose) std::cout << "  cusolverDnSgesvd(A^T): " << ev_ms << " ms\n";

    int h_info = 0;
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0)
      std::cerr << "  WARNING: cusolverDnSgesvd(A^T) info=" << h_info << "\n";

    // Step 3: d_factor (M×R col-major) = first R cols of VT^T
    //   VT (M×M col-major): VT[r,i] at vt[r + i*M].
    //   Column r of VT^T = row r of VT = left sing vec r of A.
    //   cublasSgeam: C[M×R] = (first R rows of VT)^T, treating VT as R×M with lda=M.
    CHECK_CUBLAS(cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, M, R,
      &one, d_VT, M, &zero, d_factor, M, d_factor, M));

    CHECK_CUDA(cudaFree(d_At)); CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_P));  CHECK_CUDA(cudaFree(d_VT));
    CHECK_CUDA(cudaFree(d_info)); CHECK_CUDA(cudaFree(d_work));
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaEventDestroy(ev0); cudaEventDestroy(ev1);
}


// ===================================================================
// COO tensor reader (from .tns file)
// Format: line 1 = order, line 2 = space-separated dims,
//         rest = 1-based indices + float value per non-zero
// ===================================================================
struct COOTensor {
  int order = 0;
  std::vector<uint64_t> dims;
  std::vector<std::vector<uint64_t>> indices;  // [nnz][mode], 0-based
  std::vector<float> values;
};

COOTensor readCOOTensor(const std::string& filename) {
  std::ifstream inFile(filename);
  if (!inFile) throw std::runtime_error("Cannot open COO file: " + filename);
  COOTensor coo;
  std::string line;
  bool orderRead = false, dimsRead = false;
  while (std::getline(inFile, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (!orderRead) {
      std::istringstream iss(line); iss >> coo.order;
      coo.dims.resize(coo.order); orderRead = true; continue;
    }
    if (!dimsRead) {
      std::istringstream iss(line);
      for (int i = 0; i < coo.order; i++) iss >> coo.dims[i];
      dimsRead = true; continue;
    }
    std::istringstream iss(line);
    std::vector<uint64_t> idx(coo.order);
    float val; bool ok = true;
    for (int i = 0; i < coo.order; i++) {
      if (!(iss >> idx[i])) { ok = false; break; }
      idx[i]--;  // 1-based → 0-based
    }
    if (!ok || !(iss >> val)) continue;
    coo.indices.push_back(idx);
    coo.values.push_back(val);
  }
  std::cout << "Read COO tensor: order=" << coo.order << ", nnz=" << coo.values.size() << "\n";
  return coo;
}


// ===================================================================
// CSF copy: modeOrder[0] = root (= non-contracting mode for TTMc-n),
// modeOrder[1..] = remaining modes sorted ascending by unique-index count.
// GPU pointers are null until uploadCSFToGPU() is called.
// ===================================================================
struct CSFCopy {
  int order = 3;
  std::vector<int>      modeOrder;
  std::vector<uint64_t> dims;
  std::vector<std::vector<uint64_t>> ptrs;
  std::vector<std::vector<uint64_t>> idxs;
  std::vector<float> values;

  uint64_t *d_mode0_idx = nullptr;
  uint64_t *d_mode1_ptr = nullptr, *d_mode1_idx = nullptr;
  uint64_t *d_mode2_ptr = nullptr, *d_mode2_idx = nullptr;
  float    *d_values    = nullptr;
};

CSFCopy buildCSFCopy(const COOTensor& coo, int rootMode,
                     const std::vector<size_t>& uniqueCounts) {
  const int order = coo.order;
  size_t nnz = coo.values.size();

  std::vector<int> rest;
  for (int m = 0; m < order; m++) if (m != rootMode) rest.push_back(m);
  std::sort(rest.begin(), rest.end(),
            [&](int a, int b){ return uniqueCounts[a] < uniqueCounts[b]; });

  CSFCopy csf;
  csf.order = order;
  csf.modeOrder = { rootMode, rest[0], rest[1] };
  csf.dims.resize(order);
  for (int l = 0; l < order; l++) csf.dims[l] = coo.dims[csf.modeOrder[l]];

  std::vector<size_t> perm(nnz);
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
    for (int l = 0; l < order; l++) {
      int m = csf.modeOrder[l];
      if (coo.indices[a][m] != coo.indices[b][m])
        return coo.indices[a][m] < coo.indices[b][m];
    }
    return false;
  });

  csf.ptrs.resize(order);
  csf.idxs.resize(order);
  csf.ptrs[0].push_back(0);
  std::array<uint64_t, 3> prev = { UINT64_MAX, UINT64_MAX, UINT64_MAX };

  for (size_t pi = 0; pi < nnz; pi++) {
    size_t ei = perm[pi];
    std::array<uint64_t, 3> cur = {
      coo.indices[ei][csf.modeOrder[0]],
      coo.indices[ei][csf.modeOrder[1]],
      coo.indices[ei][csf.modeOrder[2]]
    };
    bool changed = false;
    for (int l = 0; l < order; l++) {
      if (cur[l] != prev[l] || changed) {
        changed = true;
        csf.idxs[l].push_back(cur[l]);
        if (l + 1 < order) csf.ptrs[l + 1].push_back(csf.idxs[l + 1].size());
        prev[l] = cur[l];
      }
    }
    csf.values.push_back(coo.values[ei]);
  }
  for (int l = 0; l < order; l++) csf.ptrs[l].push_back(csf.idxs[l].size());
  return csf;
}

void uploadCSFToGPU(CSFCopy& csf) {
  CHECK_CUDA(cudaMalloc(&csf.d_mode0_idx, sizeof(uint64_t) * csf.idxs[0].size()));
  CHECK_CUDA(cudaMalloc(&csf.d_mode1_ptr, sizeof(uint64_t) * csf.ptrs[1].size()));
  CHECK_CUDA(cudaMalloc(&csf.d_mode1_idx, sizeof(uint64_t) * csf.idxs[1].size()));
  CHECK_CUDA(cudaMalloc(&csf.d_mode2_ptr, sizeof(uint64_t) * csf.ptrs[2].size()));
  CHECK_CUDA(cudaMalloc(&csf.d_mode2_idx, sizeof(uint64_t) * csf.idxs[2].size()));
  CHECK_CUDA(cudaMalloc(&csf.d_values,    sizeof(float)    * csf.values.size()));
  CHECK_CUDA(cudaMemcpy(csf.d_mode0_idx, csf.idxs[0].data(),
    sizeof(uint64_t) * csf.idxs[0].size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(csf.d_mode1_ptr, csf.ptrs[1].data(),
    sizeof(uint64_t) * csf.ptrs[1].size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(csf.d_mode1_idx, csf.idxs[1].data(),
    sizeof(uint64_t) * csf.idxs[1].size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(csf.d_mode2_ptr, csf.ptrs[2].data(),
    sizeof(uint64_t) * csf.ptrs[2].size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(csf.d_mode2_idx, csf.idxs[2].data(),
    sizeof(uint64_t) * csf.idxs[2].size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(csf.d_values, csf.values.data(),
    sizeof(float) * csf.values.size(), cudaMemcpyHostToDevice));
}

void freeCSFGPU(CSFCopy& csf) {
  if (csf.d_mode0_idx) { CHECK_CUDA(cudaFree(csf.d_mode0_idx)); csf.d_mode0_idx = nullptr; }
  if (csf.d_mode1_ptr) { CHECK_CUDA(cudaFree(csf.d_mode1_ptr)); csf.d_mode1_ptr = nullptr; }
  if (csf.d_mode1_idx) { CHECK_CUDA(cudaFree(csf.d_mode1_idx)); csf.d_mode1_idx = nullptr; }
  if (csf.d_mode2_ptr) { CHECK_CUDA(cudaFree(csf.d_mode2_ptr)); csf.d_mode2_ptr = nullptr; }
  if (csf.d_mode2_idx) { CHECK_CUDA(cudaFree(csf.d_mode2_idx)); csf.d_mode2_idx = nullptr; }
  if (csf.d_values)    { CHECK_CUDA(cudaFree(csf.d_values));    csf.d_values    = nullptr; }
}


// ===================================================================
// CPU reference TTMc (ncm_0 logic from v2_cpu_factorize_n_fuse.cu)
// Mirrors GPU_4L_CM_device_func_ncm_0 exactly for verification.
// arr_A: row-major (I_level1, f1), arr_B: row-major (I_level2, f2)
// arr_O: row-major (I_root, f1, f2) — must be zeroed by caller
// ===================================================================
void cpu_ttmc_ncm0(const CSFCopy& csf,
                   const float* arr_A, const float* arr_B,
                   float* arr_O, uint64_t /*I_root*/,
                   uint32_t f1, uint32_t f2) {
  std::vector<float> buffer(f2, 0.0f);
  size_t num_roots = csf.idxs[0].size();

  for (size_t i_ptr = 0; i_ptr < num_roots; ++i_ptr) {
    uint64_t i = csf.idxs[0][i_ptr];

    for (uint64_t j_ptr = csf.ptrs[1][i_ptr]; j_ptr < csf.ptrs[1][i_ptr + 1]; ++j_ptr) {
      uint64_t j = csf.idxs[1][j_ptr];

      std::fill(buffer.begin(), buffer.end(), 0.0f);

      for (uint64_t k_ptr = csf.ptrs[2][j_ptr]; k_ptr < csf.ptrs[2][j_ptr + 1]; ++k_ptr) {
        uint64_t k = csf.idxs[2][k_ptr];
        float val = csf.values[k_ptr];
        for (uint32_t s = 0; s < f2; ++s)
          buffer[s] += val * arr_B[k * f2 + s];
      }

      for (uint32_t r = 0; r < f1; ++r) {
        float a_jr = arr_A[j * f1 + r];
        for (uint32_t s = 0; s < f2; ++s)
          arr_O[i * f1 * f2 + r * f2 + s] += buffer[s] * a_jr;
      }
    }
  }
}

// Compare GPU result (arr_O_gpu, already on host) vs CPU result for mode n.
// Reports max absolute error and relative error.
void verify_ttmc(const CSFCopy& csf, const float* arr_A, const float* arr_B,
                 const float* arr_O_gpu, uint64_t I_root, uint32_t f1, uint32_t f2, int mode) {
  uint64_t size = I_root * f1 * f2;
  std::vector<float> arr_O_cpu(size, 0.0f);
  cpu_ttmc_ncm0(csf, arr_A, arr_B, arr_O_cpu.data(), I_root, f1, f2);

  float max_err = 0.0f, max_val = 0.0f;
  for (uint64_t k = 0; k < size; k++) {
    float diff = std::fabs(arr_O_gpu[k] - arr_O_cpu[k]);
    if (diff > max_err) max_err = diff;
    float v = std::fabs(arr_O_cpu[k]);
    if (v > max_val) max_val = v;
  }
  float rel_err = (max_val > 0.0f) ? max_err / max_val : max_err;
  std::cout << "[check] Mode-" << mode << " TTMc: max_abs_err=" << max_err
            << "  rel_err=" << rel_err
            << "  max_cpu_val=" << max_val << "\n";
}


int main(int argc, char* argv[]) {
  bool verbose = false;
  bool check   = false;
  g_gpu_stats  = false;
  std::string tns_file;
  std::vector<uint64_t> ranks = {10, 10, 10};
  int max_iters = 25;
  float tol = 1e-5f;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-v" || arg == "--verbose") verbose = true;
    else if (arg == "-c" || arg == "--check") check = true;
    else if (arg == "-g" || arg == "--gpu-stats") g_gpu_stats = true;
    else if ((arg == "-r" || arg == "--ranks") && i + 1 < argc) {
      ranks.clear();
      while (i + 1 < argc && argv[i + 1][0] != '-')
        ranks.push_back(static_cast<uint64_t>(atoi(argv[++i])));
    } else if ((arg == "-m" || arg == "--max-iters") && i + 1 < argc) {
      max_iters = atoi(argv[++i]);
    } else if ((arg == "-t" || arg == "--tol") && i + 1 < argc) {
      tol = static_cast<float>(atof(argv[++i]));
    } else if (tns_file.empty()) {
      tns_file = arg;
    }
  }

  if (tns_file.empty()) {
    std::cerr << "Usage: " << argv[0] << " [options] <tns_file>\n"
         << "Options:\n"
         << "  -v, --verbose         Detailed per-step timing and SVD internals\n"
         << "  -c, --check           CPU reference TTMc in iter-0 (correctness check)\n"
         << "  -g, --gpu-stats       Print size of every GPU cudaMalloc allocation\n"
         << "  -r, --ranks R0 R1 R2  Target ranks (default 10 10 10)\n"
         << "  -m, --max-iters N     Max HOOI iterations (default 25)\n"
         << "  -t, --tol T           Convergence tolerance on fit (default 1e-5)\n";
    return 1;
  }

  try {
    // ===================================================================
    // 1. Read COO tensor from .tns file
    // ===================================================================
    COOTensor coo = readCOOTensor(tns_file);
    if (coo.order != 3) {
      std::cerr << "Error: Tucker HOOI currently supports 3D tensors only (got order "
                << coo.order << ")\n";
      return 1;
    }
    while (ranks.size() < 3) ranks.push_back(10);

    uint64_t I0 = coo.dims[0], I1 = coo.dims[1], I2 = coo.dims[2];
    uint64_t R0 = std::min(ranks[0], I0);
    uint64_t R1 = std::min(ranks[1], I1);
    uint64_t R2 = std::min(ranks[2], I2);
    ranks[0] = R0; ranks[1] = R1; ranks[2] = R2;

    std::cout << "Tensor: " << I0 << " x " << I1 << " x " << I2
              << "  nnz=" << coo.values.size()
              << "  ranks=(" << R0 << "," << R1 << "," << R2 << ")\n";

    // ===================================================================
    // 2. Unique-index counts per mode (for CSF compression ordering)
    // ===================================================================
    std::vector<size_t> uniqueCounts(3);
    for (int m = 0; m < 3; m++) {
      std::unordered_set<uint64_t> uniq;
      for (const auto& idx : coo.indices) uniq.insert(idx[m]);
      uniqueCounts[m] = uniq.size();
    }
    if (verbose)
      std::cout << "Unique indices per mode: "
                << uniqueCounts[0] << ", " << uniqueCounts[1] << ", " << uniqueCounts[2] << "\n";

    // ===================================================================
    // 3. Build 3 CSF copies (mode n as root → used for TTMc-n)
    // ===================================================================
    std::vector<CSFCopy> csf_copies(3);
    for (int n = 0; n < 3; n++) {
      csf_copies[n] = buildCSFCopy(coo, n, uniqueCounts);
      if (verbose)
        std::cout << "CSF copy " << n << " (root=mode" << n << "): "
                  << "levels=[" << csf_copies[n].modeOrder[0] << ","
                  << csf_copies[n].modeOrder[1] << "," << csf_copies[n].modeOrder[2] << "]  "
                  << "roots=" << csf_copies[n].idxs[0].size()
                  << "  nnz=" << csf_copies[n].values.size() << "\n";
    }

    // ===================================================================
    // 4. Upload all CSF copies to GPU (persistent across iterations)
    // ===================================================================
    for (int n = 0; n < 3; n++) uploadCSFToGPU(csf_copies[n]);

    // ===================================================================
    // 5. Allocate and initialize factor matrices on CPU (row-major)
    // ===================================================================
    std::vector<float*> factors(3);
    std::vector<uint64_t> factor_sizes = { I0*R0, I1*R1, I2*R2 };
    for (int i = 0; i < 3; i++) {
      factors[i] = new float[factor_sizes[i]];
      init_factor_orthonormal(coo.dims[i], ranks[i], 42 + i, factors[i]);
    }

    cusolverDnHandle_t cusolverH = nullptr;
    cublasHandle_t cublasH = nullptr;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
    CHECK_CUBLAS(cublasCreate(&cublasH));

    // Warm up cusolverDnSgesvd to trigger JIT compilation before profiling.
    // not required, since I am leaving the warm up to the first iteration.of hooi
    // {
    //   const int wm = I2, wn = R0 * R1;
    //   float *d_wA, *d_wS, *d_wU, *d_wVT; int *d_winfo;
    //   CHECK_CUDA(cudaMalloc(&d_wA,   sizeof(float) * wm * wn));
    //   CHECK_CUDA(cudaMalloc(&d_wS,   sizeof(float) * wm));
    //   CHECK_CUDA(cudaMalloc(&d_wU,   sizeof(float) * wm * wm));
    //   CHECK_CUDA(cudaMalloc(&d_wVT,  sizeof(float)));
    //   CHECK_CUDA(cudaMalloc(&d_winfo,sizeof(int)));
    //   int wlwork = 0;
    //   CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(cusolverH, wm, wn, &wlwork));
    //   float *d_wwork;
    //   CHECK_CUDA(cudaMalloc(&d_wwork, sizeof(float) * std::max(wlwork, 1)));
    //   CHECK_CUSOLVER(cusolverDnSgesvd(cusolverH, 'S', 'N', wm, wn, d_wA, wm,
    //     d_wS, d_wU, wm, d_wVT, 1, d_wwork, wlwork, nullptr, d_winfo));
    //   CHECK_CUDA(cudaDeviceSynchronize());
    //   CHECK_CUDA(cudaFree(d_wA)); CHECK_CUDA(cudaFree(d_wS));
    //   CHECK_CUDA(cudaFree(d_wU)); CHECK_CUDA(cudaFree(d_wVT));
    //   CHECK_CUDA(cudaFree(d_winfo)); CHECK_CUDA(cudaFree(d_wwork));
    // }

    // Factor matrices on GPU — always kept row-major (same layout as CPU)
    float* d_factors[3];
    for (int i = 0; i < 3; i++) {
      CHECK_CUDA(cudaMalloc(&d_factors[i], sizeof(float) * factor_sizes[i]));
      CHECK_CUDA(cudaMemcpy(d_factors[i], factors[i],
        sizeof(float) * factor_sizes[i], cudaMemcpyHostToDevice));
    }

    // ===================================================================
    // 6. Output buffer sizes: arr_O_size[n] = I_n * (R product excl. R_n)
    // ===================================================================
    uint64_t arr_O_sizes[3];
    uint64_t max_arr_O_size = 0;
    for (int n = 0; n < 3; n++) {
      uint32_t f1 = static_cast<uint32_t>(ranks[csf_copies[n].modeOrder[1]]);
      uint32_t f2 = static_cast<uint32_t>(ranks[csf_copies[n].modeOrder[2]]);
      arr_O_sizes[n] = coo.dims[n] * f1 * f2;
      if (arr_O_sizes[n] > max_arr_O_size) max_arr_O_size = arr_O_sizes[n];
    }

    float* d_arr_O;
    CHECK_CUDA(cudaMalloc(&d_arr_O, sizeof(float) * max_arr_O_size));
    float* arr_O_host = allocate_aligned_array(max_arr_O_size);

    // ===================================================================
    // 7. HOOI loop
    // ===================================================================
    float prev_fit = 0.0f;
    int iter;
    double ttmc_time_us[3] = {0, 0, 0};
    double svd_time_us[3]  = {0, 0, 0};
    float input_tsr_norm = std::sqrt(
      frobenius_norm_sq_sparse(coo.values.data(), coo.values.size()));

    std::cout << "Input tensor ||X||_F = " << input_tsr_norm << "\n";
    std::cout << "Starting HOOI (max_iters=" << max_iters << ", tol=" << tol << ")\n\n";

    auto total_start = std::chrono::high_resolution_clock::now();

    for (iter = 0; iter < max_iters; iter++) {
      for (int n = 2; n >= 0; n--) {
        uint64_t arr_O_size = arr_O_sizes[n];
        CHECK_CUDA(cudaMemset(d_arr_O, 0, sizeof(float) * arr_O_size));

        // CSF copy n: mode n is root → ncm_0 outputs mode-n fibers in arr_O
        CSFCopy& csf = csf_copies[n];
        int idx_A = csf.modeOrder[1];  // level-1 (middle) mode
        int idx_B = csf.modeOrder[2];  // level-2 (leaf) mode
        uint32_t f1 = static_cast<uint32_t>(ranks[idx_A]);
        uint32_t f2 = static_cast<uint32_t>(ranks[idx_B]);

        int grid_size  = static_cast<int>(csf.idxs[0].size());
        int block_size = 1024, warp_size = 32;
        int num_warps  = (block_size + warp_size - 1) / warp_size;
        int sharedMemBytes = num_warps * f2 * sizeof(float) + f1 * f2 * sizeof(float);

        // --- GPU TTMc (ncm_0 kernel, different CSF copy per mode) ---
        auto ttmc_start = std::chrono::high_resolution_clock::now();
        GPU_4L_CM_device_func_ncm_0<<<grid_size, block_size, sharedMemBytes>>>(
          csf.d_mode0_idx,
          csf.d_mode1_ptr, csf.d_mode1_idx,
          csf.d_mode2_ptr, csf.d_mode2_idx,
          csf.d_values,
          d_factors[idx_A], d_factors[idx_B], d_arr_O,
          f1, f2, num_warps);
        CHECK_CUDA(cudaDeviceSynchronize());
        auto ttmc_end = std::chrono::high_resolution_clock::now();
        double ttmc_us = std::chrono::duration_cast<std::chrono::microseconds>(
          ttmc_end - ttmc_start).count();
        ttmc_time_us[n] += ttmc_us;
        if (verbose)
          std::cout << "[iter " << iter << " mode " << n << "] TTMc: " << ttmc_us << " us\n";

        // Copy TTMc result to host (row-major I_n × f1*f2)
        CHECK_CUDA(cudaMemcpy(arr_O_host, d_arr_O,
          sizeof(float) * arr_O_size, cudaMemcpyDeviceToHost));

        // --- Optional CPU verification (iter 0 only) ---
        if (check && iter == 0) {
          verify_ttmc(csf, factors[idx_A], factors[idx_B],
                      arr_O_host, coo.dims[n], f1, f2, n);
        }

        // --- SVD: compute top-ranks[n] left singular vectors of TTMc output ---
        uint64_t M = coo.dims[n];
        uint64_t N = (uint64_t)f1 * f2;

        // Row-major (M, N) → column-major (M, N) for cuSOLVER
        auto cm_start = std::chrono::high_resolution_clock::now();
        std::vector<float> mat_colmajor(M * N);
        for (uint64_t c = 0; c < N; c++)
          for (uint64_t r = 0; r < M; r++)
            mat_colmajor[r + c * M] = arr_O_host[r * N + c];
        if (verbose)
          std::cout << "  row→col-major: "
                    << std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::high_resolution_clock::now() - cm_start).count() << " us\n";

        float *d_mat;
        CHECK_CUDA(cudaMalloc(&d_mat, sizeof(float) * M * N));
        CHECK_CUDA(cudaMemcpy(d_mat, mat_colmajor.data(),
          sizeof(float) * M * N, cudaMemcpyHostToDevice));

        auto svd_start = std::chrono::high_resolution_clock::now();

        gpu_truncated_svd_update_factor(cusolverH, cublasH, d_mat,
          static_cast<int>(M), static_cast<int>(N),
          static_cast<int>(ranks[n]), d_factors[n], verbose);
        // gpu_full_svd_update_factor(cusolverH, cublasH, d_mat,
        //   static_cast<int>(M), static_cast<int>(N),
        //   static_cast<int>(ranks[n]), d_factors[n], verbose);
        auto svd_end = std::chrono::high_resolution_clock::now();
        double svd_us = std::chrono::duration_cast<std::chrono::microseconds>(
          svd_end - svd_start).count();
        svd_time_us[n] += svd_us;
        if (verbose)
          std::cout << "  SVD: " << svd_us << " us\n";

        // d_factors[n] is now col-major (M, R) — convert to row-major on CPU
        std::vector<float> U_host(M * ranks[n]);
        CHECK_CUDA(cudaMemcpy(U_host.data(), d_factors[n],
          sizeof(float) * M * ranks[n], cudaMemcpyDeviceToHost));
        for (uint64_t r_idx = 0; r_idx < ranks[n]; r_idx++)
          for (uint64_t i_idx = 0; i_idx < M; i_idx++)
            factors[n][i_idx * ranks[n] + r_idx] = U_host[i_idx + r_idx * M];

        // Upload row-major factor back to GPU so the kernel and the convergence GEMM
        // always see consistent row-major layout (A[i,r] = d_factors[n][i*R+r])
        CHECK_CUDA(cudaMemcpy(d_factors[n], factors[n],
          sizeof(float) * factor_sizes[n], cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaFree(d_mat));
        
      }
      if(verbose)
        std::cout << "\n--- iter " << iter << " ---\n";
        for (int n = 0; n < 3; n++) {
          std::cout << "  Mode-" << n << ": TTMc " << ttmc_time_us[n] 
                    << " us  SVD " << svd_time_us[n]  << " us\n";
          ttmc_time_us[n] = 0;
          svd_time_us[n] = 0;
        }

      // Convergence: G = A0^T × Y_ncm0 where Y = d_arr_O (last n==0 TTMc output).
      // d_factors[0] is row-major (I0, R0) = col-major (R0, I0) in cuBLAS convention.
      // GEMM: G^T (N_rest, R0) = Y^T (N_rest, I0) × A0 (I0, R0)
      uint32_t f1_0 = static_cast<uint32_t>(ranks[csf_copies[0].modeOrder[1]]);
      uint32_t f2_0 = static_cast<uint32_t>(ranks[csf_copies[0].modeOrder[2]]);
      uint64_t N_rest = (uint64_t)f1_0 * f2_0;

      float* d_G_core;
      CHECK_CUDA(cudaMalloc(&d_G_core, sizeof(float) * R0 * N_rest));
      float gemm_alpha = 1.0f, gemm_beta = 0.0f;
      CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
        N_rest, R0, I0, &gemm_alpha,
        d_arr_O,      N_rest,  // Y as col-major (N_rest, I0)
        d_factors[0], R0,      // A0 row-major (I0,R0) = cuBLAS col-major (R0,I0), OP_T → (I0,R0)
        &gemm_beta, d_G_core, N_rest));

      float* G_core_host = new float[R0 * N_rest];
      CHECK_CUDA(cudaMemcpy(G_core_host, d_G_core,
        sizeof(float) * R0 * N_rest, cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaFree(d_G_core));

      float core_norm = std::sqrt(frobenius_norm_sq_sparse(G_core_host, R0 * N_rest));
      delete[] G_core_host;

      float norm_residual = std::sqrt(
        std::max(0.0f, input_tsr_norm * input_tsr_norm - core_norm * core_norm));
      float fit = 1.0f - norm_residual / input_tsr_norm;
      float delta_fit = std::fabs(fit - prev_fit);

      // Always print per-iteration convergence info
      std::cout << "[iter " << iter << "]"
                << "  core_norm=" << core_norm
                << "  fit=" << fit
                << "  delta_fit=" << delta_fit << "\n";

      if (delta_fit < tol) {
        std::cout << "Converged (delta_fit " << delta_fit << " < tol " << tol << ")\n";
        iter++;  // so num_iters is correct
        break;
      }
      prev_fit = fit;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_us  = std::chrono::duration_cast<std::chrono::microseconds>(
      total_end - total_start).count();

    int num_iters = std::min(iter, max_iters);

    // std::cout << "\n--- Average per-iteration timing (over " << num_iters << " iterations) ---\n";
    // for (int n = 0; n < 3; n++) {
    //   std::cout << "  Mode-" << n << ": TTMc " << ttmc_time_us[n] / num_iters
    //             << " us  SVD " << svd_time_us[n] / num_iters << " us\n";
    // }

    // Final core tensor (d_arr_O still holds last ncm_0 output)
    uint32_t f1_0 = static_cast<uint32_t>(ranks[csf_copies[0].modeOrder[1]]);
    uint32_t f2_0 = static_cast<uint32_t>(ranks[csf_copies[0].modeOrder[2]]);
    uint64_t N_rest = (uint64_t)f1_0 * f2_0;

    float* d_G_final;
    CHECK_CUDA(cudaMalloc(&d_G_final, sizeof(float) * R0 * N_rest));
    {
      float a = 1.0f, b = 0.0f;
      CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
        N_rest, R0, I0, &a,
        d_arr_O,      N_rest,
        d_factors[0], R0,
        &b, d_G_final, N_rest));
    }
    float* G_core = allocate_aligned_array(R0 * N_rest);
    CHECK_CUDA(cudaMemcpy(G_core, d_G_final, sizeof(float) * R0 * N_rest, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_G_final));

    // Cleanup
    CHECK_CUDA(cudaFree(d_arr_O));
    for (int i = 0; i < 3; i++) CHECK_CUDA(cudaFree(d_factors[i]));
    for (int n = 0; n < 3; n++) freeCSFGPU(csf_copies[n]);
    std::free(arr_O_host);
    for (int i = 0; i < 3; i++) delete[] factors[i];
    std::free(G_core);
    cusolverDnDestroy(cusolverH);
    cublasDestroy(cublasH);

    std::cout << "\nTucker HOOI done: " << num_iters << " iters, " << total_us << " us total\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
