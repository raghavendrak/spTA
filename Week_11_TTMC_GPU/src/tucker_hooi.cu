/**
 * Tucker Decomposition via Higher-Order Orthogonal Iteration (HOOI)
 *
 * Implements full Tucker decomposition for 3D sparse tensors in CSF format:
 * 1. Read input tensor in CSF format
 * 2. Allocate and initialize factor matrices on CPU (random orthonormal)
 * 3. Copy tensor and factor matrices to GPU once (persistent)
 * 4. HOOI loop: TTMc (GPU kernels) + cuSOLVER SVD for each mode until convergence
 * 5. Compute core tensor, reconstruct, and verify Frobenius error
 *
 * TTMc kernels: ncm 0,1 use v7 kernels; ncm 2 uses v5 streams kernels.
 * Tensor and factor matrices are kept on GPU across iterations to avoid
 * redundant cudaMalloc/cudaMemcpy/cudaFree per iteration.
 * Usage:
 * nvcc -O3 -arch=sm_89 ./src/tucker_hooi.cu -o tucker_hooi.out -lcusolver -lcublas -lcudart
 * ./tucker_hooi.out -v -r  50 50 50  -t 0.01 -m 10  ./csf_datasets/nell-2.csf
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include "csf_tensor.h"
#include "matrix_utils.h"

#define TUCKER_HOOI_SKIP_MAIN
#include "v7_1D_grid_1D_tb_cm.cu"

// --- NCM-2 TTMc kernels (from v5_gpu_streams.cu) ---
// Part 1: accumulate values * B into buffer indexed by (k, s)
__global__ void GPU_ncm2_part1(
 const uint64_t* __restrict__ mode_1_idx,
 const uint64_t* __restrict__ mode_2_ptr, const uint64_t* __restrict__ mode_2_idx,
 const float* __restrict__ values, const float* __restrict__ arr_B,
 uint64_t f2, uint64_t j_ptr_offset,
 float* buffer_for_ncm_2, bool* k_index_buffer)
{
 uint64_t j_ptr = j_ptr_offset + blockIdx.x;
 uint64_t j = mode_1_idx[j_ptr];

 for (uint64_t k_ptr_offset = mode_2_ptr[j_ptr]; k_ptr_offset < mode_2_ptr[j_ptr + 1]; k_ptr_offset += blockDim.x) {
  uint64_t k_ptr = k_ptr_offset + threadIdx.x;
  if (k_ptr < mode_2_ptr[j_ptr + 1]) {
   float value = values[k_ptr];
   uint64_t k = mode_2_idx[k_ptr];
   if (threadIdx.y == 0) k_index_buffer[k] = true;

   for (int s_offset = 0; s_offset < (int)f2; s_offset += blockDim.y) {
    int s = s_offset + threadIdx.y;
    if (s < (int)f2) {
     uint64_t index_B = j * f2 + s;
     int buf_index = k * f2 + s;
     atomicAdd(&buffer_for_ncm_2[buf_index], value * arr_B[index_B]);
    }
   }
  }
 }
}

// Part 2: multiply buffer by A and accumulate into output
__global__ void GPU_ncm2_part2(
 const float* __restrict__ arr_A, float* arr_O,
 uint64_t f1, uint64_t f2, uint64_t i,
 const float* __restrict__ buffer_for_ncm_2, const bool* __restrict__ k_index_buffer)
{
 uint64_t k = blockIdx.x;
 if (k_index_buffer[k]) {
  for (int r_offset = 0; r_offset < (int)f1; r_offset += blockDim.y) {
   int r = r_offset + threadIdx.y;
   if (r < (int)f1) {
    float A_val = arr_A[i * f1 + r];
    for (int s_offset = 0; s_offset < (int)f2; s_offset += blockDim.x) {
     int s = s_offset + threadIdx.x;
     if (s < (int)f2) {
      uint64_t index_O = k * f1 * f2 + r * f2 + s;
      int buf_index = k * f2 + s;
      atomicAdd(&arr_O[index_O], buffer_for_ncm_2[buf_index] * A_val);
     }
    }
   }
  }
 }
}
// --- End NCM-2 kernels ---

#define CHECK_CUDA(call)                                                        \
  do {                                                                        \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
           << " \"" << cudaGetErrorString(err) << "\"\n";            \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                       \
  } while (0)

#define CHECK_CUSOLVER(call)                                                    \
  do {                                                                        \
    cusolverStatus_t status = (call);                                       \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                \
      std::cerr << "cuSOLVER error at " << __FILE__ << ":" << __LINE__    \
           << " status=" << static_cast<int>(status) << "\n";        \
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

// Initialize factor matrix with random orthonormal columns (QR of random matrix)
void init_factor_orthonormal(uint64_t rows, uint64_t cols, unsigned int seed, float* A) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (uint64_t i = 0; i < rows * cols; i++) A[i] = dist(gen);

  // Gram-Schmidt orthogonalization
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

// Compute Frobenius norm squared of sparse tensor (sum of squared values)
float frobenius_norm_sq_sparse(const float* values, size_t nnz) {
  double sum = 0;
  for (size_t i = 0; i < nnz; i++) {
    double v = values[i];
    sum += v * v;
  }
  return static_cast<float>(sum);
}

// Reconstruct single element: X_rec[i,j,k] = sum_{r,s,t} G[r,s,t] * A0[i,r] * A1[j,s] * A2[k,t]
float reconstruct_element(uint64_t i, uint64_t j, uint64_t k,
  const float* G, const float* A0, const float* A1, const float* A2,
  uint64_t R0, uint64_t R1, uint64_t R2) {
  float val = 0;
  for (uint64_t r = 0; r < R0; r++) {
    for (uint64_t s = 0; s < R1; s++) {
      float g_as = 0;
      for (uint64_t t = 0; t < R2; t++) {
        g_as += G[r * R1 * R2 + s * R2 + t] * A2[k * R2 + t];
      }
      val += A0[i * R0 + r] * A1[j * R1 + s] * g_as;
    }
  }
  return val;
}

// Compute reconstruction error at nnz positions
// For large tensors (> max_sample nnz), use first max_sample and scale for speed
void compute_frobenius_error(const CSFTensor& tensor,
  const float* G, const float* A0, const float* A1, const float* A2,
  uint64_t R0, uint64_t R1, uint64_t R2,
  float& norm_X_sq, float& norm_diff_sq) {
  const auto& ptrs = tensor.ptrs;
  const auto& idxs = tensor.idxs;
  const auto& values = tensor.values;

  norm_X_sq = 0;
  norm_diff_sq = 0;
  const uint64_t max_sample = 200000;

  uint64_t count = 0;
  for (uint64_t i_ptr = 0; i_ptr < ptrs[0][1]; i_ptr++) {
    uint64_t i = idxs[0][i_ptr];
    for (uint64_t j_ptr = ptrs[1][i_ptr]; j_ptr < ptrs[1][i_ptr + 1]; j_ptr++) {
      uint64_t j = idxs[1][j_ptr];
      for (uint64_t k_ptr = ptrs[2][j_ptr]; k_ptr < ptrs[2][j_ptr + 1]; k_ptr++) {
        uint64_t k = idxs[2][k_ptr];
        float x_val = values[k_ptr];
        float x_rec = reconstruct_element(i, j, k, G, A0, A1, A2, R0, R1, R2);
        norm_X_sq += x_val * x_val;
        float diff = x_val - x_rec;
        norm_diff_sq += diff * diff;
        if (++count >= max_sample) goto done;
      }
    }
  }
done:
  if (count < tensor.values.size() && count > 0) {
    // double scale = (double)tensor.values.size() / count;
    // norm_X_sq *= scale;
    // norm_diff_sq *= scale;
  }
}

// GPU SVD - input A is M x N (column-major), output first R cols of U to d_factor
// d_A is overwritten; d_factor should be pre-allocated M*R
void gpu_svd_update_factor(cusolverDnHandle_t cusolverH,
  float* d_A, int M, int N, int R,
  float* d_factor) {
  int min_mn = std::min(M, N);
  int lda = M;

  float *d_S = nullptr, *d_U = nullptr, *d_VT_dummy = nullptr, *d_work = nullptr;
  int *d_info = nullptr;

  CHECK_CUDA(cudaMalloc(&d_S, sizeof(float) * min_mn));
  CHECK_CUDA(cudaMalloc(&d_U, sizeof(float) * M * min_mn));
  CHECK_CUDA(cudaMalloc(&d_VT_dummy, sizeof(float)));  // jobvt='N' does not use VT
  CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

  int lwork = 0;
  CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(cusolverH, M, N, &lwork));
  CHECK_CUDA(cudaMalloc(&d_work, sizeof(float) * lwork));

  CHECK_CUSOLVER(cusolverDnSgesvd(cusolverH, 'S', 'N',
    M, N, d_A, lda, d_S, d_U, M, d_VT_dummy, 1, d_work, lwork, nullptr, d_info));

  int info;
  CHECK_CUDA(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
  if (info != 0) std::cerr << "SVD devInfo = " << info << "\n";

  // Copy first R columns of U to d_factor
  int cols_to_copy = std::min(R, min_mn);
  CHECK_CUDA(cudaMemcpy(d_factor, d_U, M * cols_to_copy * sizeof(float), cudaMemcpyDeviceToDevice));
  if (R > min_mn) {
    // Pad with zeros if R > min_mn (unusual)
    CHECK_CUDA(cudaMemset(d_factor + M * min_mn, 0, M * (R - min_mn) * sizeof(float)));
  }

  CHECK_CUDA(cudaFree(d_S));
  CHECK_CUDA(cudaFree(d_U));
  CHECK_CUDA(cudaFree(d_VT_dummy));
  CHECK_CUDA(cudaFree(d_work));
  CHECK_CUDA(cudaFree(d_info));
}

// compute_core_gpu removed: core tensor is now computed directly via
// cuBLAS GEMM on persistent GPU data (G = A0^T × TTMc_ncm0_output)

int main(int argc, char* argv[]) {
  bool verbose = false;
  std::string csf_file;
  std::vector<uint64_t> ranks = {10, 10, 10};
  int max_iters = 25;
  float tol = 1e-5f;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-v" || arg == "--verbose") verbose = true;
    else if ((arg == "-r" || arg == "--ranks") && i + 1 < argc) {
      ranks.clear();
      while (i + 1 < argc && argv[i + 1][0] != '-') {
        ranks.push_back(static_cast<uint64_t>(atoi(argv[++i])));
      }
    } else if ((arg == "-m" || arg == "--max-iters") && i + 1 < argc) {
      max_iters = atoi(argv[++i]);
    } else if ((arg == "-t" || arg == "--tol") && i + 1 < argc) {
      tol = static_cast<float>(atof(argv[++i]));
    } else if (csf_file.empty()) {
      csf_file = arg;
    }
  }

  if (csf_file.empty()) {
    std::cerr << "Usage: " << argv[0] << " [options] <csf_file>\n"
         << "Options:\n"
         << "  -v, --verbose       Verbose output\n"
         << "  -r, --ranks R0 R1 R2  Target ranks (default 10 10 10)\n"
         << "  -m, --max-iters N    Max HOOI iterations (default 25)\n"
         << "  -t, --tol T          Convergence tolerance (default 1e-5)\n";
    return 1;
  }

  try {
    CSFTensor tensor = readCSFTensor(csf_file);
    if (tensor.order != 3) {
      std::cerr << "Error: Tucker HOOI currently supports 3D tensors only (got order " << tensor.order << ")\n";
      return 1;
    }
    while (ranks.size() < 3) ranks.push_back(10);

    uint64_t I0 = tensor.dimensions[0], I1 = tensor.dimensions[1], I2 = tensor.dimensions[2];
    uint64_t R0 = ranks[0], R1 = ranks[1], R2 = ranks[2];
    R0 = std::min(R0, I0); R1 = std::min(R1, I1); R2 = std::min(R2, I2);
    ranks[0] = R0; ranks[1] = R1; ranks[2] = R2;

    if (verbose) {
      std::cout << "Tensor dimensions: " << I0 << " x " << I1 << " x " << I2 << "\n";
      std::cout << "Target ranks: " << R0 << ", " << R1 << ", " << R2 << "\n";
      std::cout << "Nonzeros: " << tensor.values.size() << "\n";
    }

    std::vector<uint64_t*> mode_ptrs, mode_idxs;
    float* values;
    int order;
    getCSFArrays(tensor, mode_ptrs, mode_idxs, values, order);

    std::vector<size_t> size_mode_ptr(3), size_mode_idx(3);
    for (int i = 0; i < 3; i++) {
      size_mode_ptr[i] = tensor.ptrs[i].size();
      size_mode_idx[i] = tensor.idxs[i].size();
    }

    // 2. Allocate and initialize factor matrices on CPU
    std::vector<float*> factors(3);
    std::vector<uint64_t> factor_sizes(3);
    for (int i = 0; i < 3; i++) {
      factors[i] = new float[tensor.dimensions[i] * ranks[i]];
      init_factor_orthonormal(tensor.dimensions[i], ranks[i], 42 + i, factors[i]);
      factor_sizes[i] = tensor.dimensions[i] * ranks[i];
    }

    cusolverDnHandle_t cusolverH = nullptr;
    cublasHandle_t cublasH = nullptr;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
    CHECK_CUBLAS(cublasCreate(&cublasH));

    // ===================================================================
    // 3. Persistent GPU allocation: tensor CSF arrays + factor matrices
    // ===================================================================
    uint64_t total_values = size_mode_idx[2];

    // CSF tensor arrays on GPU (copied once, never changes)
    uint64_t *d_mode_0_idx, *d_mode_1_ptr, *d_mode_1_idx, *d_mode_2_ptr, *d_mode_2_idx;
    float *d_values;
    CHECK_CUDA(cudaMalloc(&d_mode_0_idx, sizeof(uint64_t) * size_mode_idx[0]));
    CHECK_CUDA(cudaMalloc(&d_mode_1_ptr, sizeof(uint64_t) * size_mode_ptr[1]));
    CHECK_CUDA(cudaMalloc(&d_mode_1_idx, sizeof(uint64_t) * size_mode_idx[1]));
    CHECK_CUDA(cudaMalloc(&d_mode_2_ptr, sizeof(uint64_t) * size_mode_ptr[2]));
    CHECK_CUDA(cudaMalloc(&d_mode_2_idx, sizeof(uint64_t) * size_mode_idx[2]));
    CHECK_CUDA(cudaMalloc(&d_values, sizeof(float) * total_values));

    CHECK_CUDA(cudaMemcpy(d_mode_0_idx, mode_idxs[0], sizeof(uint64_t) * size_mode_idx[0], cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mode_1_ptr, mode_ptrs[1], sizeof(uint64_t) * size_mode_ptr[1], cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mode_1_idx, mode_idxs[1], sizeof(uint64_t) * size_mode_idx[1], cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mode_2_ptr, mode_ptrs[2], sizeof(uint64_t) * size_mode_ptr[2], cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mode_2_idx, mode_idxs[2], sizeof(uint64_t) * size_mode_idx[2], cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, values, sizeof(float) * total_values, cudaMemcpyHostToDevice));

    // Factor matrices on GPU (updated after each SVD)
    float* d_factors[3];
    for (int i = 0; i < 3; i++) {
      CHECK_CUDA(cudaMalloc(&d_factors[i], sizeof(float) * factor_sizes[i]));
      CHECK_CUDA(cudaMemcpy(d_factors[i], factors[i], sizeof(float) * factor_sizes[i], cudaMemcpyHostToDevice));
    }

    // Pre-compute output sizes for each mode
    uint64_t arr_O_sizes[3];
    uint64_t max_arr_O_size = 0;
    for (int n = 0; n < 3; n++) {
      arr_O_sizes[n] = 1;
      for (int i = 0; i < 3; i++) {
        if (i != n) arr_O_sizes[n] *= ranks[i];
        else arr_O_sizes[n] *= tensor.dimensions[i];
      }
      if (arr_O_sizes[n] > max_arr_O_size) max_arr_O_size = arr_O_sizes[n];
    }

    // Persistent output buffer on GPU (reused across modes/iterations)
    float* d_arr_O;
    CHECK_CUDA(cudaMalloc(&d_arr_O, sizeof(float) * max_arr_O_size));

    // NCM-2 work buffers (buffer indexed by k*f2+s, and boolean k-index tracker)
    // For ncm=2: idx_A=0, idx_B=1, f2=R1, buffer size = I2 * R1
    float* d_buffer_ncm2 = nullptr;
    bool* d_kindex_ncm2 = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buffer_ncm2, sizeof(float) * I2 * R1));
    CHECK_CUDA(cudaMalloc(&d_kindex_ncm2, sizeof(bool) * I2));

    // ===================================================================
    // 4. HOOI loop with per-mode timing
    // ===================================================================
    float prev_fit = 0.0f;
    int iter;
    double ttmc_time_ms[3] = {0, 0, 0};
    double svd_time_ms[3] = {0, 0, 0};
    float input_tsr_norm = std::sqrt(frobenius_norm_sq_sparse( tensor.values.data(), tensor.values.size()));

    auto total_start = std::chrono::high_resolution_clock::now();

    for (iter = 0; iter < max_iters; iter++) {
      for (int n = 2; n >= 0; n--) {
        uint64_t arr_O_size = arr_O_sizes[n];
        CHECK_CUDA(cudaMemset(d_arr_O, 0, sizeof(float) * arr_O_size));

        int idx_A, idx_B;
        if (n == 0) { idx_A = 1; idx_B = 2; }
        else if (n == 1) { idx_A = 0; idx_B = 2; }
        else { idx_A = 0; idx_B = 1; }
        uint32_t f1 = ranks[idx_A];
        uint32_t f2 = ranks[idx_B];

        // --- TTMc ---
        auto ttmc_start = std::chrono::high_resolution_clock::now();

        if (n == 0) {
          int grid_size = static_cast<int>(size_mode_idx[0]);
          int block_size = 1024, warp_size = 32;
          int num_warps = (block_size + warp_size - 1) / warp_size;
          int sharedMemBytes = num_warps * f2 * sizeof(float) + f1 * f2 * sizeof(float);

          GPU_4L_CM_device_func_ncm_0<<<grid_size, block_size, sharedMemBytes>>>(
            d_mode_0_idx, d_mode_1_ptr, d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
            d_values, d_factors[idx_A], d_factors[idx_B], d_arr_O, f1, f2, num_warps);
          CHECK_CUDA(cudaDeviceSynchronize());
        } else if (n == 1) {
          int grid_size = static_cast<int>(size_mode_idx[0]);
          int block_size = 1024, warp_size = 32;
          int num_warps = (block_size + warp_size - 1) / warp_size;
          int sharedMemBytes = num_warps * f2 * sizeof(float);

          GPU_4L_CM_device_func_ncm_1<<<grid_size, block_size, sharedMemBytes>>>(
            d_mode_0_idx, d_mode_1_ptr, d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
            d_values, d_factors[idx_A], d_factors[idx_B], d_arr_O, f1, f2, num_warps);
          CHECK_CUDA(cudaDeviceSynchronize());
        } else {
          // ncm == 2: loop over mode-0 fibers, launch part1+part2 per fiber
          for (uint64_t i_ptr = 0; i_ptr < mode_ptrs[0][1]; ++i_ptr) {
            uint64_t i_val = mode_idxs[0][i_ptr];
            uint64_t j_ptr_offset = mode_ptrs[1][i_ptr];
            uint64_t num_j = mode_ptrs[1][i_ptr + 1] - mode_ptrs[1][i_ptr];

            CHECK_CUDA(cudaMemset(d_buffer_ncm2, 0, sizeof(float) * I2 * f2));
            CHECK_CUDA(cudaMemset(d_kindex_ncm2, 0, sizeof(bool) * I2));

            dim3 blockDim_ncm2(32, 32);

            dim3 gridDim1(num_j);
            GPU_ncm2_part1<<<gridDim1, blockDim_ncm2>>>(
              d_mode_1_idx, d_mode_2_ptr, d_mode_2_idx,
              d_values, d_factors[idx_B],
              (uint64_t)f2, j_ptr_offset,
              d_buffer_ncm2, d_kindex_ncm2);

            dim3 gridDim2(I2);
            GPU_ncm2_part2<<<gridDim2, blockDim_ncm2>>>(
              d_factors[idx_A], d_arr_O,
              (uint64_t)f1, (uint64_t)f2, i_val,
              d_buffer_ncm2, d_kindex_ncm2);
          }
          CHECK_CUDA(cudaDeviceSynchronize());
        }

        auto ttmc_end = std::chrono::high_resolution_clock::now();
        ttmc_time_ms[n] += std::chrono::duration<double, std::milli>(ttmc_end - ttmc_start).count();

        // --- SVD: update factor matrix n ---
        auto svd_start = std::chrono::high_resolution_clock::now();

        // Copy TTMc output from GPU to CPU for row-to-column-major conversion
        float* arr_O_host = allocate_aligned_array(arr_O_size);
        CHECK_CUDA(cudaMemcpy(arr_O_host, d_arr_O, sizeof(float) * arr_O_size, cudaMemcpyDeviceToHost));

        uint64_t M = tensor.dimensions[n];
        uint64_t N = (n == 0) ? (R1 * R2) : (n == 1) ? (R0 * R2) : (R0 * R1);

        // Row-major (M, N) -> column-major (M, N) for cuSOLVER
        std::vector<float> mat_colmajor(M * N);
        for (uint64_t c = 0; c < N; c++)
          for (uint64_t r = 0; r < M; r++)
            mat_colmajor[r + c * M] = arr_O_host[r * N + c];

        float *d_mat, *d_factor;
        CHECK_CUDA(cudaMalloc(&d_mat, sizeof(float) * M * N));
        CHECK_CUDA(cudaMalloc(&d_factor, sizeof(float) * M * ranks[n]));
        CHECK_CUDA(cudaMemcpy(d_mat, mat_colmajor.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice));

        gpu_svd_update_factor(cusolverH, d_mat, static_cast<int>(M), static_cast<int>(N),
          static_cast<int>(ranks[n]), d_factor);

        // U is column-major (M, R); convert to row-major and update CPU factor
        std::vector<float> U_host(M * ranks[n]);
        CHECK_CUDA(cudaMemcpy(U_host.data(), d_factor, sizeof(float) * M * ranks[n], cudaMemcpyDeviceToHost));
        for (uint64_t r_idx = 0; r_idx < ranks[n]; r_idx++)
          for (uint64_t i_idx = 0; i_idx < M; i_idx++)
            factors[n][i_idx * ranks[n] + r_idx] = U_host[i_idx + r_idx * M];

        // Update the persistent GPU factor matrix
        CHECK_CUDA(cudaMemcpy(d_factors[n], factors[n], sizeof(float) * factor_sizes[n], cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaFree(d_mat));
        CHECK_CUDA(cudaFree(d_factor));
        std::free(arr_O_host);

        auto svd_end = std::chrono::high_resolution_clock::now();
        svd_time_ms[n] += std::chrono::duration<double, std::milli>(svd_end - svd_start).count();
      }

      // Convergence: compute core G = A0^T × Y via cuBLAS GEMM
      // d_arr_O holds TTMc ncm=0 result Y (row-major I0 × R1*R2) = col-major (R1*R2, I0)^T
      // d_factors[0] holds updated A0 (row-major I0 × R0) = col-major (R0, I0)^T
      // G^T = Y^T × A0 => col-major (R1*R2, R0) = row-major (R0, R1*R2)
      float* d_G_core;
      CHECK_CUDA(cudaMalloc(&d_G_core, sizeof(float) * R0 * R1 * R2));
      float gemm_alpha = 1.0f, gemm_beta = 0.0f;
      CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
        R1 * R2, R0, I0, &gemm_alpha,
        d_arr_O, R1 * R2,       // Y^T col-major (R1*R2, I0)
        d_factors[0], R0,        // A0^T col-major (R0, I0), transposed to A0
        &gemm_beta, d_G_core, R1 * R2));  // G^T col-major (R1*R2, R0) = G row-major

      float* G_core_host = new float[R0 * R1 * R2];
      CHECK_CUDA(cudaMemcpy(G_core_host, d_G_core, sizeof(float) * R0 * R1 * R2, cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaFree(d_G_core));

      float core_tsr_norm = std::sqrt(frobenius_norm_sq_sparse(G_core_host, R0 * R1 * R2));
      delete[] G_core_host;
      float norm_residual = std::sqrt(input_tsr_norm * input_tsr_norm - core_tsr_norm * core_tsr_norm);
      float fit = 1 - norm_residual / input_tsr_norm;

      std::cout << "input_tsr_norm: " << input_tsr_norm << ", core_tsr_norm: " << core_tsr_norm << ", fit: " << fit << "\n";
      if (
        // (prev_fit >= 0 && std::fabs(fit - prev_fit) < tol) ||//no change in fit
      (std::fabs(fit - prev_fit) < tol)) { // or fit is less than tol
        if (verbose) std::cout << "Converged at iteration " << iter + 1 << " (fit change < tol)\n";
        break;
      }
      prev_fit = fit;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

    // Number of iterations that actually ran
    int num_iters = std::min(iter + 1, max_iters);

    // Print per-mode average timings
    std::cout << "\n--- Average per-iteration timing (over " << num_iters << " iterations) ---\n";
    for (int n = 0; n < 3; n++) {
      std::cout << "Mode-" << n << " TTMc: " << ttmc_time_ms[n] / num_iters
           << " ms, SVD: " << svd_time_ms[n] / num_iters << " ms\n";
    }
    std::cout << "\n";

    // Final core tensor: G = A0^T × Y using d_arr_O and d_factors[0] still on GPU
    float* d_G_final;
    CHECK_CUDA(cudaMalloc(&d_G_final, sizeof(float) * R0 * R1 * R2));
    {
      float a = 1.0f, b = 0.0f;
      CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
        R1 * R2, R0, I0, &a,
        d_arr_O, R1 * R2,
        d_factors[0], R0,
        &b, d_G_final, R1 * R2));
    }
    float* G_core = allocate_aligned_array(R0 * R1 * R2);
    CHECK_CUDA(cudaMemcpy(G_core, d_G_final, sizeof(float) * R0 * R1 * R2, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_G_final));

    // Free persistent GPU memory
    CHECK_CUDA(cudaFree(d_arr_O));
    CHECK_CUDA(cudaFree(d_buffer_ncm2));
    CHECK_CUDA(cudaFree(d_kindex_ncm2));
    for (int i = 0; i < 3; i++) CHECK_CUDA(cudaFree(d_factors[i]));
    CHECK_CUDA(cudaFree(d_mode_0_idx));
    CHECK_CUDA(cudaFree(d_mode_1_ptr));
    CHECK_CUDA(cudaFree(d_mode_1_idx));
    CHECK_CUDA(cudaFree(d_mode_2_ptr));
    CHECK_CUDA(cudaFree(d_mode_2_idx));
    CHECK_CUDA(cudaFree(d_values));

    // 5. Reconstruct and verify Frobenius error
    float norm_X_sq, norm_diff_sq;
    compute_frobenius_error(tensor, G_core, factors[0], factors[1], factors[2],
      R0, R1, R2, norm_X_sq, norm_diff_sq);

    float norm_X = std::sqrt(norm_X_sq);
    float norm_diff = std::sqrt(norm_diff_sq);
    float rel_error = (norm_X > 1e-10f) ? (norm_diff / norm_X) : norm_diff;

    std::cout << "Tucker HOOI completed in " << num_iters << " iterations, " << total_ms << " ms\n";
    std::cout << "Frobenius norm of input: " << norm_X << "\n";
    std::cout << "Reconstruction error ||X - X_rec||_F: " << norm_diff << "\n";
    std::cout << "Relative Frobenius error: " << rel_error << "\n";

    if (verbose) {
      std::cout << "Fit (||G||_F^2): " << prev_fit << "\n";
    }

    // Cleanup
    for (int i = 0; i < 3; i++) delete[] factors[i];
    std::free(G_core);
    cusolverDnDestroy(cusolverH);
    cublasDestroy(cublasH);

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
